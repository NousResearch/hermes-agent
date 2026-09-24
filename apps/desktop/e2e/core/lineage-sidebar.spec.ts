/**
 * Session lineage and sidebar integrity after compaction and branching.
 * One real Electron app + one real `hermes serve`; only the LLM is faked.
 *
 *  - compaction that rotates the session (compression.in_place: false) is a
 *    CONTINUATION: the sidebar keeps one row for the conversation, never a
 *    parent + child branch pair (#121148);
 *  - the prompt acknowledged right after compaction renders exactly once,
 *    live and after a reload (#121088);
 *  - a branch child is born with a title (#121062) and is a sidebar row of
 *    its own;
 *  - switching between the branch and its parent never renders the other
 *    session's (stale) turns or duplicates a part (#121096).
 *
 * Known bugs are strict expected failures (KNOWN below): the test turns red
 * the moment the bug is fixed, forcing the entry out.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  renderedTranscript,
  send,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { installDuplicateSampler } from './oracle'
import { startScriptedProvider } from './provider'
import { sessionRows } from './remote-helpers'

/** Scenario → issue for bugs confirmed on main; test.fail() while listed. */
const KNOWN: Record<string, string> = {}

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** Visible sidebar session rows (rows own a [data-row-actions] cluster; chat bubbles do not). */
function sidebarRows(page: Page) {
  const row = '*:has(> [data-row-actions])'

  return page.locator(`${row}:not(${row} *)`).filter({ visible: true })
}

async function sidebarRowTexts(page: Page): Promise<string[]> {
  return (await sidebarRows(page).allInnerTexts()).map(text => text.replace(/\s+/g, ' ').trim())
}

/** Markers of every user bubble, in order, and how often each assistant marker renders. */
async function renderedMarkers(page: Page): Promise<{ users: string[]; assistants: string[] }> {
  const { bubbles } = await renderedTranscript(page)
  const pick = (role: string, re: RegExp) =>
    bubbles.filter(b => b.role === role).flatMap(b => b.text.match(re) ?? [])

  return {
    users: pick('user', new RegExp(`\\bU\\d+-${nonce}\\b`, 'g')),
    assistants: pick('assistant', new RegExp(`\\bA\\d+-${nonce}\\b`, 'g'))
  }
}

const isSummary = (c: { body: any }) => JSON.stringify(c.body?.messages ?? '').includes('context checkpoint')

async function composeSlash(page: Page, command: string) {
  const box = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await box.click()
  await page.keyboard.insertText(command)
  await expect.poll(() => box.textContent()).toContain(command.slice(1))
  await page.getByRole('button', { name: 'Send', exact: true }).click()
}

async function settled(page: Page) {
  await expect
    .poll(() => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').count(), {
      timeout: 60_000,
      message: 'turn settled'
    })
    .toBe(0)
}

async function openSession(page: Page, sessionId: string, mustShow: string) {
  await page.evaluate(id => {
    window.location.hash = `#/${encodeURIComponent(id)}`
  }, sessionId)
  await expect.poll(() => currentSessionId(page)).toBe(sessionId)
  await expect(viewport(page)).toContainText(mustShow, { timeout: 60_000 })
}

test('lineage: compaction continuation and branch children keep one coherent sidebar', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('lineage')
  writeProviderHome(
    sandbox.hermesHome,
    provider.url,
    'compression:\n  in_place: false\n  protect_first_n: 1\n  protect_last_n: 1\n'
  )
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  const finished = (marker: string) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker}`
      })
      .toBe(true)

  const turn = async (n: number, words: string) => {
    provider.script(U(n), [{ text: [`${A(n)} `, ...words.split(' ').map(w => `${w} `)] }])
    await send(page, `${U(n)} ${words}`, 'Enter', ws)
    await finished(U(n))
    await expect(viewport(page)).toContainText(A(n))
    await settled(page)
  }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    let rootId = ''

    await test.step('compaction continuation stays one sidebar row (#121148)', async () => {
      if (KNOWN.continuation) {
        test.info().annotations.push({ type: 'known', description: KNOWN.continuation })
      }

      await turn(1, 'first question here')
      rootId = storedSessionForMarker(sandbox, 'default', U(1)) ?? ''
      expect(rootId).not.toBe('')
      await turn(2, 'second question here')
      await turn(3, 'third question here')
      await expect.poll(() => sidebarRows(page).count()).toBe(1)

      const summariesBefore = provider.completions.filter(isSummary).length
      await composeSlash(page, '/compress keep the question markers')
      await expect
        .poll(() => provider.completions.filter(c => isSummary(c) && c.finished).length, {
          timeout: 90_000,
          message: 'the compaction summary was generated'
        })
        .toBeGreaterThan(summariesBefore)
      await settled(page)
      await turn(4, 'after the compaction')
      const continuation = storedSessionForMarker(sandbox, 'default', U(4))
      expect(continuation, 'follow-up persisted').not.toBeNull()
      const rotated = continuation !== rootId
      test.info().annotations.push({ type: 'compaction', description: rotated ? 'rotated' : 'in place' })

      if (rotated) {
        expect(sessionRows(sandbox).find(r => r.id === continuation)?.parent_session_id).toBe(rootId)
      }

      await expect
        .poll(() => sidebarRowTexts(page), {
          timeout: 30_000,
          message: 'one conversation → one sidebar row after compaction'
        })
        .toHaveLength(1)
    })

    await test.step('the prompt acknowledged after compaction renders once (#121088)', async () => {
      const live = await renderedMarkers(page)
      expect(live.users.filter(m => m === U(4)), 'live: U4 rendered once').toHaveLength(1)
      expect(live.assistants.filter(m => m === A(4)), 'live: A4 rendered once').toHaveLength(1)
      expect(live.users.indexOf(U(4)), 'U4 is the last user bubble').toBe(live.users.length - 1)

      await page.reload()
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await expect(viewport(page)).toContainText(A(4), { timeout: 60_000 })
      const cold = await renderedMarkers(page)
      expect(cold.users.filter(m => m === U(4)), 'reload: U4 rendered once').toHaveLength(1)
      expect(cold.assistants.filter(m => m === A(4)), 'reload: A4 rendered once').toHaveLength(1)
    })

    let branchId = ''
    let parentId = ''

    await test.step('a branch child is titled and is its own row (#121062)', async () => {
      parentId = await currentSessionId(page)
      const before = new Set(sessionRows(sandbox).map(r => r.id))
      const row = sidebarRows(page).first()
      await row.click({ button: 'right' })
      await page.getByRole('menuitem', { name: /^branch/i }).first().click()
      await expect
        .poll(() => sessionRows(sandbox).filter(r => !before.has(r.id)).length, {
          timeout: 60_000,
          message: 'the branch created a session row'
        })
        .toBeGreaterThan(0)
      await expect.poll(() => currentSessionId(page), { timeout: 60_000 }).not.toBe(parentId)
      await turn(5, 'only on the branch')
      branchId = storedSessionForMarker(sandbox, 'default', U(5)) ?? ''
      expect(branchId).not.toBe('')
      expect(before.has(branchId), 'U5 landed in a NEW session').toBe(false)

      await expect
        .poll(() => sessionRows(sandbox).find(r => r.id === branchId)?.title ?? null, {
          timeout: 30_000,
          message: 'branch child has a title'
        })
        .not.toBeNull()
      await expect.poll(() => sidebarRows(page).count(), { timeout: 30_000 }).toBe(2)
    })

    await test.step('switching branch ↔ parent never renders stale or duplicated parts (#121096)', async () => {
      for (let round = 0; round < 3; round++) {
        await openSession(page, parentId, A(4))
        const parent = await renderedMarkers(page)
        expect(parent.users, `round ${round}: parent never shows the branch turn`).not.toContain(U(5))
        expect(new Set(parent.assistants).size, `round ${round}: parent assistant parts unique`).toBe(
          parent.assistants.length
        )

        await openSession(page, branchId, A(5))
        const branch = await renderedMarkers(page)
        expect(branch.users.filter(m => m === U(5)), `round ${round}: branch turn once`).toHaveLength(1)
        expect(new Set(branch.assistants).size, `round ${round}: branch assistant parts unique`).toBe(
          branch.assistants.length
        )
      }

      await page.reload()
      await waitForInteractive(app, page)
      await expect.poll(() => sidebarRows(page).count(), { timeout: 30_000 }).toBe(2)
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
