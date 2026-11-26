/**
 * #145333: a running turn stores its prompt and each finished tool round as it goes. In a chat
 * longer than one page, reopening it mid-turn merged the newest page, which already holds those
 * rows, beside the live stream's copy of the turn: the prompt drew twice, the finished tools under
 * the first copy and the running reply under the second, until a reload.
 *
 * The tab loads the chat from the store first (as a freshly opened tab does), so the newest page
 * at the reopen starts inside a turn the window holds as one fold, and the merge pairs rows by
 * stored id. Real gateway, real Webapp renderer; only the model is scripted. Linux, like the other
 * Webapp journeys; needs `pnpm run build:webapp` and a Chromium Playwright can launch.
 */

import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { expect, type Page, test } from '@playwright/test'

import {
  composer,
  type CoreSandbox,
  currentSessionId,
  recordWebSockets,
  renderedTranscript,
  send,
  webappLaunchLink,
  withWebapp
} from './harness'
import { gate } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-reopen${nonce}`
const A = (n: number) => `A${n}-reopen${nonce}`
const N = (n: number, k: number) => `A${n}i${k}-reopen${nonce}`

const words = (marker: string, ...rest: string[]) => [`${marker} `, ...rest.map(word => `${word} `)]

// Each round is a narration row and its tool rows, so a page cut inside a turn starts at a later fold.
const SEED_TURNS = 3
const SEED_ROUNDS = 4
const ROUND_CALLS = 12
const R = SEED_TURNS + 1

const todos = (tag: string, count: number) =>
  Array.from({ length: count }, (_, i) => ({
    name: 'todo_list',
    args: { todos: [{ id: `${tag}-${i}`, content: `${tag} item ${i}`, status: 'pending' }] }
  }))

/** True once the profile's state.db holds an assistant row containing `text`. */
function assistantStored(sandbox: CoreSandbox, text: string): boolean {
  const db = new DatabaseSync(path.join(sandbox.hermesHome, 'state.db'), { readOnly: true })

  try {
    return Boolean(
      db.prepare("SELECT 1 FROM messages WHERE role = 'assistant' AND content LIKE ? LIMIT 1").get(`%${text}%`)
    )
  } catch {
    return false // a read mid-WAL-checkpoint; the caller polls
  } finally {
    db.close()
  }
}

/** How many rendered bubbles carry each marker. */
async function bubbleCounts(page: Page, markers: string[]): Promise<number[]> {
  const { bubbles } = await renderedTranscript(page)

  return markers.map(marker => bubbles.filter(bubble => bubble.text.includes(marker)).length)
}

/** Samples over `ms` where a marker is drawn in fewer than `least` bubbles or more than one. */
async function drawnOutside(page: Page, markers: string[], ms: number, least: number): Promise<string[]> {
  const bad: string[] = []
  const deadline = Date.now() + ms

  while (Date.now() < deadline) {
    const counts = await bubbleCounts(page, markers)

    if (counts.some(count => count < least || count > 1)) {
      bad.push(markers.map((marker, i) => `${marker}=${counts[i]}`).join(' '))
    }

    await page.waitForTimeout(250)
  }

  return bad
}

test('a long Webapp chat reopened mid-turn draws the running turn once', async () => {
  test.setTimeout(600_000)
  const hold = gate()

  await withWebapp('webapp-running-reopen', async (sandbox, provider, browser) => {
    try {
      const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage()
      const ws = recordWebSockets(page)
      const viewport = page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
      await page.goto(webappLaunchLink(sandbox))
      await expect(composer(page)).toBeVisible({ timeout: 180_000 })

      await test.step(`${SEED_TURNS} narrated tool turns push the newest page past the chat start`, async () => {
        for (let n = 1; n <= SEED_TURNS; n += 1) {
          provider.script(U(n), [
            ...Array.from({ length: SEED_ROUNDS }, (_, k) => ({
              text: words(N(n, k + 1), 'seeding'),
              toolCalls: todos(`seed-${n}-${k}`, ROUND_CALLS)
            })),
            { text: words(A(n), 'seeded') }
          ])
          await send(page, `${U(n)} seed the chat`, 'Enter', ws)
          await expect(viewport).toContainText(A(n), { timeout: 120_000 })
        }
      })

      const sessionId = await currentSessionId(page)
      const prompt = [U(R)]
      const rounds = [N(R, 1), N(R, 2)]

      await test.step('the tab loads the chat from the store', async () => {
        await page.reload()
        await expect(viewport).toContainText(A(SEED_TURNS), { timeout: 120_000 })
      })

      await test.step('the next turn stores its prompt and two tool rounds, then holds its answer', async () => {
        provider.script(U(R), [
          { text: words(N(R, 1), 'first', 'look'), toolCalls: todos('look-1', ROUND_CALLS) },
          { text: words(N(R, 2), 'second', 'look'), toolCalls: todos('look-2', ROUND_CALLS) },
          { text: words(A(R), 'answer', 'still', 'running'), holdAfterFirstChunk: hold }
        ])
        await send(page, `${U(R)} check twice`, 'Enter', ws)
        await provider.streamStarted(U(R), 2)
        await expect(viewport).toContainText(A(R))
        await expect.poll(() => assistantStored(sandbox, N(R, 2)), { timeout: 60_000 }).toBe(true)
      })

      await test.step('reopening the chat mid-turn never draws the prompt twice', async () => {
        await page.evaluate(() => {
          window.location.hash = '#/'
        })
        await expect(viewport).not.toContainText(U(R))
        await page.evaluate(id => {
          window.location.hash = `#/${encodeURIComponent(id)}`
        }, sessionId)
        await expect(viewport).toContainText(U(R))

        expect((await drawnOutside(page, prompt, 6_000, 0)).slice(0, 5)).toEqual([])
        await expect.poll(() => bubbleCounts(page, [...prompt, ...rounds]), { timeout: 15_000 }).toEqual([1, 1, 1])
        expect((await drawnOutside(page, [...prompt, ...rounds], 3_000, 1)).slice(0, 5)).toEqual([])
      })

      await test.step('the turn settles with every part drawn once', async () => {
        hold.open()
        await expect(viewport).toContainText('answer still running', { timeout: 60_000 })
        await expect
          .poll(() =>
            ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(A(R)))
          )
          .toBe(true)

        expect((await drawnOutside(page, [...prompt, ...rounds, A(R)], 4_000, 1)).slice(0, 5)).toEqual([])
      })
    } finally {
      hold.open()
    }
  })
})
