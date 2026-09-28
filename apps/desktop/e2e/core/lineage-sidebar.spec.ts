/**
 * Session lineage and sidebar integrity for branches. One real Electron app +
 * one real `hermes serve`; only the LLM is faked.
 *
 *  - a branch child is born with a title (#121062) and is a sidebar row of
 *    its own;
 *  - switching between the branch and its parent (3 round trips, then a
 *    reload) never renders the other session's turns or duplicates a part.
 *
 * Compaction lineages live in lineage-rotation.spec.ts (#121148) and
 * lineage-compaction-prompt.spec.ts (#121088). The #121096 mechanism
 * (index-keyed response-group children) is NOT reached by this switching
 * scenario — see the PR's NOT COVERED list.
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

/** Markers of every user bubble, in order, and how often each assistant marker renders. */
async function renderedMarkers(page: Page): Promise<{ users: string[]; assistants: string[] }> {
  const { bubbles } = await renderedTranscript(page)

  const pick = (role: string, re: RegExp) => bubbles.filter(b => b.role === role).flatMap(b => b.text.match(re) ?? [])

  return {
    users: pick('user', new RegExp(`\\bU\\d+-${nonce}\\b`, 'g')),
    assistants: pick('assistant', new RegExp(`\\bA\\d+-${nonce}\\b`, 'g'))
  }
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

test('lineage: a branch child is its own titled row and switching never leaks turns', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('lineage')
  writeProviderHome(sandbox.hermesHome, provider.url)
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

    await test.step('a parent conversation with two turns', async () => {
      await turn(1, 'first question here')
      await turn(2, 'second question here')
      await expect.poll(() => sidebarRows(page).count()).toBe(1)
    })

    let branchId = ''
    let parentId = ''

    await test.step('a branch child is titled and is its own row (#121062)', async () => {
      parentId = await currentSessionId(page)
      const before = new Set(sessionRows(sandbox).map(r => r.id))
      const row = sidebarRows(page).first()
      await row.click({ button: 'right' })
      await page
        .getByRole('menuitem', { name: /^branch/i })
        .first()
        .click()
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

    await test.step("switching branch ↔ parent never renders the other session's turn or a duplicate part", async () => {
      for (let round = 0; round < 3; round++) {
        await openSession(page, parentId, A(2))
        const parent = await renderedMarkers(page)
        expect(parent.users, `round ${round}: parent never shows the branch turn`).not.toContain(U(5))
        expect(new Set(parent.assistants).size, `round ${round}: parent assistant parts unique`).toBe(
          parent.assistants.length
        )

        await openSession(page, branchId, A(5))
        const branch = await renderedMarkers(page)
        expect(
          branch.users.filter(m => m === U(5)),
          `round ${round}: branch turn once`
        ).toHaveLength(1)
        expect(new Set(branch.assistants).size, `round ${round}: branch assistant parts unique`).toBe(
          branch.assistants.length
        )
      }

      // Every frame sampled across the switches (transient duplicates included).
      const violations = await page.evaluate(() => (window as any).__coreSampler?.violations ?? [])
      expect(violations, 'no marker rendered twice in any sampled frame').toEqual([])

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
