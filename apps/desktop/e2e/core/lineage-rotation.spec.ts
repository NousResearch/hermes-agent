/**
 * #121148 with REAL rotated session rows: one Electron app + one `hermes serve`,
 * only the LLM is faked. `compression.in_place: false` + a low
 * `threshold_tokens` make the backend's own auto-compaction rotate the live
 * conversation mid-turn (parent sealed with end_reason='compression', a
 * continuation row with parent_session_id = parent) — verified from state.db
 * before anything is asserted about the sidebar.
 *
 * The sidebar runs in Project grouping (the persisted preference the grouping
 * menu writes). The sandbox holds exactly one conversation, so every visible
 * session row belongs to that one compression lineage.
 *
 *  - KNOWN #121148 (last assertion, merge-order safe): while the conversation is live, the continuation renders
 *    as a `└─` branch row nested under its sealed parent (same shape as a
 *    user branch) — observed in a bounded window around the rotating turn.
 *  - Plain regression: after a reload the lineage is ONE sidebar row, it is
 *    the live tip (clicking it routes to the tip id from state.db) and it
 *    opens the continuation transcript.
 *
 * Not covered here: collapseCompressionLineages (session-branch-tree.ts). No
 * real sequence tried (manual /compress, mid-turn auto-compaction, flat and
 * Project grouping, pinned root, reload) ever put two rows with the same
 * lineage key into the sidebar pool — the backend tip projection,
 * mergeSessionPage's lineage dedupe and upsertResolvedSession already remove
 * them — so sabotaging it does not turn this spec red.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { expectNoSymptom } from './known'
import { startScriptedProvider } from './provider'
import { messageRows, type SessionRow, sessionRows } from './remote-helpers'

const KNOWN: Record<string, string> = {
  continuationAsBranch:
    '#121148 a compression continuation renders as a nested └─ branch row under its sealed parent (Project grouping)'
}

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

/** Bulky assistant prose: compaction only shrinks the transcript (and commits) when the replaced turns outweigh the summary wrapper. */
const bulk = (tag: string) => Array.from({ length: 1500 }, (_, i) => `${tag}w${i}${'abcdefghij'[i % 10]}`).join(' ')

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

const ROW = '*:has(> [data-row-actions])'

/** Visible sidebar SESSION rows (outermost [data-row-actions] owners that name this run's conversation, not project headers). */
function sessionRowsInSidebar(page: Page) {
  return page
    .locator(`${ROW}:not(${ROW} *)`)
    .filter({ visible: true })
    .filter({ hasText: new RegExp(`-${nonce}\\b|Untitled`) })
}

/** In-page MutationObserver: the most session rows ever rendered at once, and their texts. */
async function installRowCountSampler(page: Page) {
  await page.evaluate(
    ({ row, nonce }) => {
      const re = new RegExp(`-${nonce}\\b|Untitled`)

      const rec = { max: 0, atMax: [] as string[], samples: 0 }

      ;(window as unknown as { __rotationRows: typeof rec }).__rotationRows = rec

      const sample = () => {
        rec.samples++

        const texts = [...document.querySelectorAll<HTMLElement>(`${row}:not(${row} *)`)]
          .filter(el => el.offsetParent !== null)
          .map(el => el.innerText.replace(/\s+/g, ' ').trim())
          .filter(text => re.test(text))

        if (texts.length > rec.max) {
          rec.max = texts.length
          rec.atMax = texts.map(text => text.slice(0, 60))
        }
      }

      new MutationObserver(sample).observe(document.body, {
        attributes: true,
        characterData: true,
        childList: true,
        subtree: true
      })
      sample()
    },
    { row: ROW, nonce }
  )
}

async function readRowCountSampler(page: Page): Promise<{ max: number; atMax: string[]; samples: number }> {
  return page.evaluate(
    () => (window as unknown as { __rotationRows: { max: number; atMax: string[]; samples: number } }).__rotationRows
  )
}

async function settled(page: Page) {
  await expect
    .poll(() => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').count(), {
      timeout: 60_000,
      message: 'turn settled'
    })
    .toBe(0)
}

/** The live tip of `rootId`'s compression lineage, walking state.db's compression edges. */
function lineageFromDb(rows: SessionRow[], rootId: string): string[] {
  const chain = [rootId]

  for (;;) {
    const cur = rows.find(r => r.id === chain[chain.length - 1])
    const next = cur?.end_reason === 'compression' ? rows.find(r => r.parent_session_id === cur.id) : undefined

    if (!next) {
      return chain
    }

    chain.push(next.id)
  }
}

test('lineage: a real compression rotation is one sidebar row per conversation (#121148)', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('rotation')
  writeProviderHome(
    sandbox.hermesHome,
    provider.url,
    'compression:\n  in_place: false\n  protect_first_n: 1\n  protect_last_n: 1\n  threshold_tokens: 24000\n'
  )
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  const turn = async (n: number, words: string) => {
    provider.script(U(n), [{ text: [`${A(n)} `, `${bulk(`a${n}`)} `, 'done'] }])
    await send(page, `${U(n)} ${words}`, 'Enter', ws)
    await expect(viewport(page)).toContainText(A(n), { timeout: 60_000 })
    await settled(page)
  }

  try {
    await waitForInteractive(app, page)
    // Project grouping, as the sidebar's grouping menu persists it.
    await page.evaluate(() => localStorage.setItem('hermes.desktop.agentsGroupedByWorkspace', 'true'))
    await page.reload()
    await waitForInteractive(app, page)

    let rootId = ''
    let last = 0

    await test.step('auto-compaction really rotates the live conversation (state.db)', async () => {
      await turn(1, 'first question')
      rootId = storedSessionForMarker(sandbox, 'default', U(1)) ?? ''
      expect(rootId, 'U1 persisted').not.toBe('')
      await expect(page.locator('[data-sessions-mode]').first()).toHaveAttribute('data-sessions-mode', 'projects', {
        timeout: 30_000
      })
      await expect(sessionRowsInSidebar(page)).toHaveCount(1)
      await installRowCountSampler(page)

      for (last = 2; last <= 9; last++) {
        await turn(last, `question number ${last}`)

        if (sessionRows(sandbox).some(r => r.id === rootId && r.end_reason === 'compression')) {
          break
        }
      }

      const rows = sessionRows(sandbox)
      expect(rows.find(r => r.id === rootId)?.end_reason, 'root sealed by compression').toBe('compression')
      const continuation = rows.find(r => r.parent_session_id === rootId)
      expect(continuation, 'continuation row with parent_session_id = root').toBeTruthy()
    })

    let seen = { max: 0, atMax: [] as string[], samples: 0 }

    await test.step('observe the live sidebar around the rotation (#121148 window)', async () => {
      // Bounded observation window (not synchronization): the rotating turn,
      // 3 s idle, the follow-up turn (which normally rotates again), 3 s idle.
      await page.waitForTimeout(3_000)
      await turn(last + 1, 'after the rotation')
      await page.waitForTimeout(3_000)
      seen = await readRowCountSampler(page)
      expect(seen.samples, 'row sampler observed the sidebar').toBeGreaterThan(0)
    })

    await test.step('after reload the lineage is one row: the live tip, opening the continuation', async () => {
      const tip = lineageFromDb(sessionRows(sandbox), rootId).at(-1)!
      expect(tip, 'rotation moved the tip off the root').not.toBe(rootId)
      expect(
        messageRows(sandbox, tip).some(m => m.role === 'user' && m.content.includes(U(last + 1))),
        'follow-up persisted on the tip'
      ).toBe(true)

      await page.reload()
      await waitForInteractive(app, page)
      await expect
        .poll(async () => (await sessionRowsInSidebar(page).allInnerTexts()).map(t => t.replace(/\s+/g, ' ').trim()), {
          timeout: 30_000,
          message: 'one conversation → one sidebar row after rotation + reload'
        })
        .toHaveLength(1)

      // Record every route the click produces: a stale root row would route to
      // the sealed root first (the backend then redirects the resume to the tip).
      await page.evaluate(() => {
        window.location.hash = '#/'
      })
      await expect.poll(() => currentSessionId(page)).toBe('')
      await page.evaluate(() => {
        const w = window as unknown as { __routes: string[] }
        w.__routes = []

        const record = () => {
          const id = decodeURIComponent(location.hash.replace(/^#\/?/, '').split('?')[0] ?? '')

          if (w.__routes.at(-1) !== id) {
            w.__routes.push(id)
          }
        }

        for (const method of ['pushState', 'replaceState'] as const) {
          const original = history[method].bind(history)

          history[method] = (...args: Parameters<History['pushState']>) => {
            original(...args)
            record()
          }
        }

        window.addEventListener('hashchange', record)
      })
      await sessionRowsInSidebar(page).first().click()
      await expect
        .poll(() => page.evaluate(() => (window as unknown as { __routes: string[] }).__routes.filter(Boolean)), {
          timeout: 30_000,
          message: 'the sidebar row IS the live tip: clicking it routes straight to the tip id'
        })
        .toEqual([tip])
      await expect(viewport(page)).toContainText(A(last + 1), { timeout: 60_000 })
      await expect(sessionRowsInSidebar(page)).toHaveCount(1)
    })

    // Last: the KNOWN symptom (an expected failure while #121148 is open, a pass once fixed).
    expectNoSymptom(
      KNOWN.continuationAsBranch,
      seen.max > 1,
      'one conversation rendered as more than one sidebar row after a compression rotation',
      `max ${seen.max} rows at once: ${JSON.stringify(seen.atMax)}`
    )
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
