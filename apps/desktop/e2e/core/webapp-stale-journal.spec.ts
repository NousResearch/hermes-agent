/**
 * A Webapp tab that dies while a reply streams keeps that turn in its in-flight journal
 * (localStorage). On the next open, the owner's pending list (the reopen snapshot's admissions
 * not yet finished) decides what the journal is worth:
 *  - the turn finished: the journal is stale. The session moved on from another device, a
 *    tool-heavy turn pushed the old prompt out of the newest transcript page, and a newer turn
 *    completed, all within minutes. The old turn must not be replayed below the newest reply,
 *    and the journal is retired so the next open is clean too.
 *  - the gateway died with the turn (the restarted owner lists it `unknown`): the journal holds
 *    the only copy of the streamed reply, and the reopened chat shows it.
 *
 * Real gateway, real Webapp renderer, real browser storage; only the model is scripted. Linux, like
 * the other Webapp journeys; needs `npm run build:webapp` and a Chromium Playwright can launch.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { type Browser, type BrowserContext, chromium, expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  composer,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  freePort,
  recordWebSockets,
  renderedTranscript,
  sandboxProcesses,
  send,
  storedSessionForMarker,
  webappLaunchLink,
  writeProviderHome
} from './harness'
import { gate, type Gate, type ScriptedProvider, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-journal${nonce}`
const A = (n: number) => `A${n}-journal${nonce}`

function readState<T>(sandbox: CoreSandbox, read: (db: DatabaseSync) => T, fallback: T): T {
  const db = new DatabaseSync(path.join(sandbox.hermesHome, 'state.db'), { readOnly: true })

  try {
    return read(db)
  } catch {
    return fallback // a read mid-WAL-checkpoint; the caller polls
  } finally {
    db.close()
  }
}

/** True once the profile's state.db holds an assistant row containing `text`. */
const assistantPersisted = (sandbox: CoreSandbox, text: string) =>
  readState(
    sandbox,
    db =>
      Boolean(
        db.prepare("SELECT 1 FROM messages WHERE role = 'assistant' AND content LIKE ? LIMIT 1").get(`%${text}%`)
      ),
    false
  )

/** Statuses of the session's admissions in the owner's ledger, oldest first. */
const admissionStatuses = (sandbox: CoreSandbox, sessionId: string) =>
  readState(
    sandbox,
    db =>
      (
        db.prepare('SELECT status FROM session_admissions WHERE target_session_id = ? ORDER BY seq').all(sessionId) as {
          status: string
        }[]
      ).map(row => row.status),
    [] as string[]
  )

interface JournalEntry {
  messages: { role: string; parts: { type: string; text?: string }[] }[]
}

/** A stored value that is a live journal entry (a retired one is removed or tombstoned). */
function parseJournalEntry(raw: null | string | undefined): JournalEntry | null {
  try {
    const entry = raw ? JSON.parse(raw) : null

    return Array.isArray(entry?.messages) ? entry : null
  } catch {
    return null
  }
}

/** The session's journal key: found by what the tab stored, not spelled out, so a journal version bump cannot
 *  leave the retirement checks below reading a key nothing writes. */
const journalKeyFor = (sessionId: string) => (key: string) =>
  key.includes('inflightTurnJournal') && key.endsWith(`:${encodeURIComponent(sessionId)}`)

async function readJournal(page: Page, sessionId: string): Promise<{ key: string; entry: JournalEntry } | null> {
  const stored = await page.evaluate(() => Object.entries(localStorage))
  const isKey = journalKeyFor(sessionId)

  for (const [key, raw] of stored) {
    const entry = isKey(key) ? parseJournalEntry(raw) : null

    if (entry) {
      return { key, entry }
    }
  }

  return null
}

const journalText = (entry: JournalEntry | null, role: string) =>
  (entry?.messages ?? [])
    .filter(message => message.role === role)
    .flatMap(message => message.parts.map(part => part.text ?? ''))
    .join(' ')

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** Bubbles of the old turn (its prompt or its reply) rendered after the newest prompt. */
async function oldTurnAfterNewestPrompt(page: Page): Promise<string[]> {
  const { bubbles } = await renderedTranscript(page)
  const newest = bubbles.findLastIndex(bubble => bubble.role === 'user' && bubble.text.includes(U(3)))

  if (newest < 0) {
    return []
  }

  return bubbles
    .slice(newest + 1)
    .filter(bubble => bubble.text.includes(U(1)) || bubble.text.includes(A(1)))
    .map(bubble => `${bubble.role}: ${bubble.text.slice(0, 80)}`)
}

interface Phone {
  context: BrowserContext
  sessionId: string
  /** Whether the phone's localStorage still holds a live journal for the session. */
  journalStored: () => Promise<boolean>
}

/** The phone signs in, sends a prompt, and its tab dies while the reply streams. */
async function phoneDiesMidTurn(browser: Browser, sandbox: CoreSandbox, provider: ScriptedProvider, hold: Gate) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 860 } })
  const tab = await context.newPage()
  const ws = recordWebSockets(tab)
  await tab.goto(webappLaunchLink(sandbox))
  await expect(composer(tab)).toBeVisible({ timeout: 180_000 })

  provider.script(U(1), [{ text: [`${A(1)} `, 'the ', 'old ', 'turn'], holdAfterFirstChunk: hold }])
  await send(tab, `${U(1)} start the old turn`, 'Enter', ws)
  await provider.streamStarted(U(1))
  await expect.poll(() => currentSessionId(tab)).not.toBe('')
  const sessionId = await currentSessionId(tab)
  const origin = new URL(tab.url()).origin
  await expect.poll(() => storedSessionForMarker(sandbox, 'default', U(1))).toBe(sessionId)

  // The journal holds the prompt and the streamed start of the reply.
  await expect
    .poll(
      async () => {
        const entry = (await readJournal(tab, sessionId))?.entry ?? null

        return {
          prompt: journalText(entry, 'user').includes(U(1)),
          reply: journalText(entry, 'assistant').includes(A(1))
        }
      },
      { timeout: 60_000, message: 'the tab journaled the streaming turn' }
    )
    .toEqual({ prompt: true, reply: true })

  const { key } = (await readJournal(tab, sessionId))!
  await tab.close()

  const journalStored = async () =>
    ((await context.storageState()).origins.find(entry => entry.origin === origin)?.localStorage ?? []).some(
      item => item.name === key && parseJournalEntry(item.value) !== null
    )

  expect(await journalStored(), 'the dead tab left its journal behind').toBe(true)

  return { context, sessionId, journalStored } satisfies Phone
}

/** A new tab on the phone opens the chat: its sign-in grant lives in the tab's own
 *  sessionStorage, so it signs in with a launch link first. `newestPage` is the body of the
 *  first newest-page transcript read the reopened renderer makes for the chat. */
async function phoneReopens(phone: Phone, sandbox: CoreSandbox): Promise<{ tab: Page; newestPage: Promise<string> }> {
  const tab = await phone.context.newPage()
  await tab.goto(webappLaunchLink(sandbox))
  await expect(composer(tab)).toBeVisible({ timeout: 180_000 })

  const newestPage = tab
    .waitForResponse(
      response =>
        response.url().includes(`/sessions/${encodeURIComponent(phone.sessionId)}/messages?`) &&
        response.url().includes('order=latest'),
      { timeout: 60_000 }
    )
    .then(response => response.text())
    .catch(() => '')

  await tab.evaluate(id => {
    window.location.hash = `#/${encodeURIComponent(id)}`
  }, phone.sessionId)
  await expect.poll(() => currentSessionId(tab)).toBe(phone.sessionId)

  return { tab, newestPage }
}

async function withWebapp(
  label: string,
  body: (sandbox: CoreSandbox, provider: ScriptedProvider, browser: Browser) => Promise<void>
) {
  test.skip(process.platform !== 'linux', 'runs with the other Webapp journeys (Linux CI)')
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox(label)
  writeProviderHome(sandbox.hermesHome, provider.url)
  // todo_list is deferred behind tool_search by default; the scripted model calls it directly.
  fs.appendFileSync(
    path.join(sandbox.hermesHome, 'config.yaml'),
    `\nwebapp:\n  enabled: true\n  port: ${await freePort()}\ntools:\n  tool_search:\n    defer: []\n`
  )
  let browser: Browser | null = null

  try {
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    await body(sandbox, provider, browser)
  } finally {
    await browser?.close().catch(() => undefined)

    for (const proc of sandboxProcesses(sandbox)) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
}

test('a Webapp tab that died mid-turn does not replay that turn after the newest reply', async () => {
  test.setTimeout(480_000)
  const hold = gate()

  await withWebapp('webapp-journal', async (sandbox, provider, browser) => {
    try {
      const phone = await phoneDiesMidTurn(browser, sandbox, provider, hold)

      // The turn finishes on the gateway with no tab watching it.
      hold.open()
      await expect.poll(() => assistantPersisted(sandbox, `${A(1)} the old turn`), { timeout: 120_000 }).toBe(true)

      // Another device, minutes later: a tool-heavy turn pushes the old prompt out of the newest
      // page, then a newer turn completes.
      const other = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
      const otherWs = recordWebSockets(other)
      await other.goto(webappLaunchLink(sandbox))
      await expect(composer(other)).toBeVisible({ timeout: 180_000 })
      await other.evaluate(id => {
        window.location.hash = `#/${encodeURIComponent(id)}`
      }, phone.sessionId)
      await expect.poll(() => currentSessionId(other)).toBe(phone.sessionId)
      await expect(viewport(other)).toContainText(`${A(1)} the old turn`, { timeout: 60_000 })

      // Distinct calls, so no repeated-call guard collapses them; 4 × (1 + 32) rows outnumber the newest page.
      // todo_list runs in-process, so the rounds cost no shell spawns.
      const tools = (step: number) =>
        Array.from({ length: 32 }, (_, i) => ({
          name: 'todo_list',
          args: { todos: [{ id: `${step}-${i}`, content: `step ${step} item ${i}`, status: 'pending' }] }
        }))

      provider.script(U(2), [
        { toolCalls: tools(0) },
        { toolCalls: tools(1) },
        { toolCalls: tools(2) },
        { toolCalls: tools(3) },
        { text: [`${A(2)} `, 'ran ', 'the ', 'tools'] }
      ])
      await send(other, `${U(2)} run the tools`, 'Enter', otherWs)
      await expect.poll(() => assistantPersisted(sandbox, `${A(2)} ran the tools`), { timeout: 240_000 }).toBe(true)

      provider.script(U(3), [{ text: [`${A(3)} `, 'the ', 'newest ', 'reply'] }])
      await send(other, `${U(3)} the newest prompt`, 'Enter', otherWs)
      await expect.poll(() => assistantPersisted(sandbox, `${A(3)} the newest reply`), { timeout: 120_000 }).toBe(true)
      await expect(viewport(other)).toContainText(`${A(3)} the newest reply`, { timeout: 60_000 })

      const { tab, newestPage } = await phoneReopens(phone, sandbox)
      await expect(viewport(tab)).toContainText(`${A(3)} the newest reply`, { timeout: 60_000 })

      // The newest page the reopened renderer hydrated with no longer reaches the old prompt.
      const pageText = await newestPage
      expect(pageText, 'the newest page holds the newest turn').toContain(U(3))
      expect(pageText, 'the old prompt is older than the newest page').not.toContain(U(1))

      // Resume folds the journal: either it retires it, or it replays the old turn at the end.
      await expect
        .poll(async () => !(await phone.journalStored()) || (await oldTurnAfterNewestPrompt(tab)).length > 0, {
          timeout: 60_000,
          message: 'the reopened tab settled its journal'
        })
        .toBe(true)
      expect(await oldTurnAfterNewestPrompt(tab), 'the old turn is not shown after the newest prompt').toEqual([])
      expect(await phone.journalStored(), 'the journal was retired').toBe(false)

      // A second open is clean as well.
      await tab.reload()
      await expect(composer(tab)).toBeVisible({ timeout: 120_000 })
      await expect(viewport(tab)).toContainText(`${A(3)} the newest reply`, { timeout: 60_000 })
      const { bubbles } = await renderedTranscript(tab)
      expect(bubbles.findLast(bubble => bubble.role === 'user')?.text).toContain(U(3))
      expect(await oldTurnAfterNewestPrompt(tab)).toEqual([])
    } finally {
      hold.open()
    }
  })
})

test('a Webapp tab whose gateway died mid-turn still shows the streamed reply after a restart', async () => {
  test.setTimeout(480_000)
  const hold = gate()

  await withWebapp('webapp-journal-crash', async (sandbox, provider, browser) => {
    try {
      const phone = await phoneDiesMidTurn(browser, sandbox, provider, hold)

      // The gateway dies with the turn: the streamed reply was never stored.
      const [gateway] = backendProcesses(sandbox)
      expect(gateway, 'one gateway serves the sandbox').toBeTruthy()
      process.kill(gateway.pid, 'SIGKILL')
      hold.open()
      expect(assistantPersisted(sandbox, A(1)), 'the streamed reply exists only in the journal').toBe(false)

      // The restarted owner fences the turn it lost as `unknown` and keeps listing it.
      const { tab } = await phoneReopens(phone, sandbox)
      await expect.poll(() => admissionStatuses(sandbox, phone.sessionId), { timeout: 60_000 }).toEqual(['unknown'])

      await expect(viewport(tab)).toContainText(`${U(1)} start the old turn`, { timeout: 60_000 })
      await expect(viewport(tab)).toContainText(A(1), { timeout: 60_000 })
      expect(await phone.journalStored(), 'the journal is kept until the lost turn is resolved').toBe(true)
    } finally {
      hold.open()
    }
  })
})
