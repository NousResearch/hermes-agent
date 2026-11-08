/**
 * Stop mid-stream: the reply on screen is the reply the session saved (#121594).
 *
 * Stop seals the live bubble at the click and the renderer drops every later
 * message.delta, but the agent keeps streaming until it honours the interrupt
 * and persists everything it delivered — state.db and the model's next-turn
 * context then hold words the user never saw. The interrupted message.complete
 * carries that persisted partial; the bubble must extend to it.
 *
 * Each run streams numbered words (`wNNN`) from the scripted provider, presses
 * the real Stop button mid-stream, waits for the interrupted completion on the
 * wire and the assistant row in state.db, then compares the last word rendered
 * with the last word persisted. Red on base: the bubble was 3-5 words short at
 * the completion in every run; the first chat of a launch stayed short for
 * good (later chats were repaired only by an unrelated transcript re-read
 * ~0.4-0.7 s later). HERMES_E2E_INTERRUPT_RUNS / _OUT drive the N-run A/B.
 */

import fs from 'node:fs'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const RUNS = Number(process.env.HERMES_E2E_INTERRUPT_RUNS || 2)
const WORDS = 400
const STOP_AFTER = 15
const word = (i: number) => `w${String(i).padStart(3, '0')}`

const lastWord = (text: string) => {
  const all = text.match(/\bw(\d{3})\b/g) ?? []

  return all.length ? Number(all.at(-1)!.slice(1)) : -1
}

function persistedReply(dbPath: string, marker: string): null | string {
  if (!fs.existsSync(dbPath)) {
    return null
  }

  const db = new DatabaseSync(dbPath, { readOnly: true })

  try {
    const row = db
      .prepare("SELECT content FROM messages WHERE role = 'assistant' AND content LIKE ? ORDER BY id DESC LIMIT 1")
      .get(`%${marker}%`) as undefined | { content: string }

    return row?.content ?? null
  } catch {
    return null
  } finally {
    db.close()
  }
}

async function renderedReply(page: Page, marker: string): Promise<string> {
  return page.evaluate(marker => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    const bubbles = [...(viewport?.querySelectorAll('[data-slot="aui_assistant-message-root"]') ?? [])]

    return bubbles
      .map(el => (el as HTMLElement).innerText)
      .filter(text => text.includes(marker))
      .join('\n')
  }, marker)
}

test('Stop mid-stream renders exactly the partial reply the session persisted', async () => {
  test.setTimeout(120_000 + RUNS * 45_000)
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('interrupt')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const dbPath = path.join(sandbox.hermesHome, 'state.db')
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  const results: {
    run: number
    atStop: number
    atComplete: number
    healMs: number
    wire: number
    rendered: number
    persisted: number
    status: string
  }[] = []

  try {
    await waitForInteractive(app, page)

    for (let run = 1; run <= RUNS; run++) {
      if (run > 1) {
        await page.evaluate(() => {
          window.location.hash = '#/'
        })
        await expect.poll(() => currentSessionId(page)).toBe('')
      }

      provider.script(U(run), [{ text: [`${A(run)} `, ...Array.from({ length: WORDS }, (_, i) => `${word(i + 1)} `)] }])
      await send(page, `${U(run)} count`, 'Enter', ws)
      await provider.streamStarted(U(run))
      await page.waitForFunction(
        ([marker, needle]) =>
          [...document.querySelectorAll('[data-slot="aui_assistant-message-root"]')].some(
            el => (el as HTMLElement).innerText.includes(marker) && (el as HTMLElement).innerText.includes(needle)
          ),
        [A(run), word(STOP_AFTER)] as const,
        { polling: 'raf', timeout: 60_000 }
      )
      await page
        .locator('[data-slot="composer-root"] button[aria-label="Stop"]')
        .filter({ visible: true })
        .first()
        .click()
      const atStop = lastWord(await renderedReply(page, A(run)))

      const findComplete = () =>
        ws.events.find(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(A(run)))

      await expect
        .poll(() => Boolean(findComplete()), { intervals: [20], message: `completion for ${U(run)}` })
        .toBe(true)
      const completeAt = Date.now()
      const complete = findComplete()!
      const atComplete = lastWord(await renderedReply(page, A(run)))

      await expect.poll(() => persistedReply(dbPath, A(run)), { message: `assistant row for ${U(run)}` }).not.toBeNull()
      const persisted = lastWord(persistedReply(dbPath, A(run))!)
      // Bounded observation window: how long after the completion the screen
      // first matches state.db (-1: never). A dropped tail never heals from
      // the completion alone, so polling cannot mask it.
      let rendered = -1
      let healMs = -1

      await expect
        .poll(
          async () => {
            rendered = lastWord(await renderedReply(page, A(run)))
            healMs = rendered === persisted ? Date.now() - completeAt : -1

            return rendered
          },
          { timeout: 5_000, intervals: [20] }
        )
        .toBe(persisted)
        .catch(() => undefined)

      const streamed = ws.events
        .filter(e => e.type === 'message.delta' && e.sessionId === complete.sessionId)
        .map(e => String(e.payload?.text ?? ''))
        .join('')

      results.push({
        run,
        atStop,
        atComplete,
        healMs,
        wire: lastWord(streamed.slice(streamed.lastIndexOf(A(run)))),
        rendered,
        persisted,
        status: String(complete.payload?.status ?? '')
      })
    }
  } finally {
    const out = process.env.HERMES_E2E_INTERRUPT_OUT

    if (out) {
      fs.writeFileSync(out, JSON.stringify(results, null, 2))
    }

    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }

  // Stop must land mid-stream for the run to mean anything.
  expect(results.every(r => r.status === 'interrupted' && r.persisted < WORDS)).toBe(true)
  expect(results.filter(r => r.rendered !== r.persisted)).toEqual([])
})
