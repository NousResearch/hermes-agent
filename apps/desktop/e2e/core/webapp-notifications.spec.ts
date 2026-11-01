/**
 * Webapp notifications where `new Notification()` throws, as on mobile browsers.
 *
 * Chrome for Android and most other mobile browsers throw a TypeError from the Notification
 * constructor and show only what a service worker registration shows. In Chromium that refusal is the
 * Blink runtime feature NotificationConstructor, off only on Android; desktop Chromium launched with
 * it disabled refuses with Android's own message. This is desktop Chromium, not Android. Everything
 * else is real: the gateway serves the Webapp and its notification worker, an approval raised by a
 * real turn in a background chat is the attention event, and the worker's registration shows the
 * notification, as does Settings' "Send test notification".
 *
 * Playwright cannot click an OS notification. The click is the real worker's `notificationclick`
 * handler run on the notification it showed, as a synthetic event dispatched inside the worker. It
 * proves the routing back to the session; it does not prove the browser's own delivery of a click or
 * that the tab takes focus (a synthetic click may not focus a window).
 *
 * Needs the Webapp renderer (`npm run build:webapp`) and a Chromium Playwright can launch
 * (`npx playwright install chromium`, or HERMES_E2E_CHROMIUM).
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import { createServer } from 'node:net'
import * as path from 'node:path'

import { type Browser, chromium, expect, type Page, test } from '@playwright/test'

import {
  composer,
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  writeProviderHome
} from './harness'
import { shellQuote } from './process-census'
import { gate, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-notify${nonce}`
const A = (n: number) => `A${n}-notify${nonce}`
const AI = (n: number) => `A${n}i-notify${nonce}`

const PYTHON = process.env.HERMES_E2E_PYTHON || 'python3'

const MOBILE_REFUSAL =
  "Failed to construct 'Notification': Illegal constructor. Use ServiceWorkerRegistration.showNotification() instead."

interface ShownNotification {
  body: string
  focusSessionId: unknown
  tag: string
  title: string
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      server.close(() => resolve(typeof address === 'object' && address ? address.port : 0))
    })
  })
}

/** Starts the gateway if needed and returns a one-use launch link for its Webapp. */
function launchLink(sandbox: CoreSandbox): string {
  const result = spawnSync(PYTHON, ['-m', 'hermes_cli.main', 'webapp', '--no-open', '--skip-build'], {
    cwd: REPO_ROOT,
    env: coreAppEnv(sandbox),
    encoding: 'utf8',
    timeout: 180_000
  })

  const output = `${result.stdout ?? ''}${result.stderr ?? ''}`
  const link = /^\s+(http:\/\/127\.0\.0\.1:\d+\/webapp\/launch#\S+)$/m.exec(output)?.[1]
  expect(result.status, `hermes webapp --no-open\n${output.replace(/ticket=\S+/g, 'ticket=<redacted>')}`).toBe(0)
  expect(link, 'hermes webapp printed a one-use launch link').toBeTruthy()

  return link!
}

type NotifyDoor = (payload: { kind?: string }) => Promise<boolean>

/**
 * Record what `hermesDesktop.notify`, the door every attention event goes through, answers: a
 * pass-through spy, so a failure names the outcome (on the unfixed Webapp, the constructor's
 * TypeError) instead of only an empty notification list.
 */
function recordNotifyOutcomes(page: Page): Promise<void> {
  return page.evaluate(() => {
    const host = window as unknown as { __notifyOutcomes: string[]; hermesDesktop: { notify: NotifyDoor } }
    const notify = host.hermesDesktop.notify
    host.__notifyOutcomes = []
    host.hermesDesktop.notify = payload =>
      notify(payload).then(
        shown => {
          host.__notifyOutcomes.push(`${payload.kind}: ${shown}`)

          return shown
        },
        (error: unknown) => {
          host.__notifyOutcomes.push(`${payload.kind}: ${String(error)}`)
          throw error
        }
      )
  })
}

function notifyOutcomes(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __notifyOutcomes: string[] }).__notifyOutcomes)
}

/** What the notification worker's registration shows right now, as the page sees it. */
function shownNotifications(page: Page): Promise<ShownNotification[]> {
  return page.evaluate(async () => {
    const registration = await navigator.serviceWorker.getRegistration()
    const shown = registration ? await registration.getNotifications() : []

    return shown.map(notification => ({
      body: notification.body,
      focusSessionId: (notification.data as { focusSessionId?: unknown } | null)?.focusSessionId,
      tag: notification.tag,
      title: notification.title
    }))
  })
}

test('a Webapp notification shows through its worker where the constructor throws, and its click opens the chat', async () => {
  test.skip(process.platform !== 'linux', 'runs with the other Webapp journeys (Linux CI)')
  test.setTimeout(300_000)
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp-notify')
  writeProviderHome(sandbox.hermesHome, provider.url, '', 'manual')
  const port = await freePort()
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nwebapp:\n  enabled: true\n  port: ${port}\n`)
  const hold = gate()
  let browser: Browser | null = null

  try {
    // Full Chromium in its new headless mode: the old headless shell denies notifications whatever is
    // granted. The constructor is refused as on Android (see the header).
    const args = ['--disable-blink-features=NotificationConstructor']

    browser = await chromium.launch(
      process.env.HERMES_E2E_CHROMIUM
        ? { args, executablePath: process.env.HERMES_E2E_CHROMIUM }
        : { args, channel: 'chromium' }
    )

    const context = await browser.newContext({
      permissions: ['notifications'],
      viewport: { width: 1280, height: 860 }
    })

    const page = await context.newPage()
    const ws = recordWebSockets(page)
    await page.goto(launchLink(sandbox))
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await recordNotifyOutcomes(page)

    expect(
      await page.evaluate(() => {
        try {
          new Notification('probe')

          return `shown (${Notification.permission})`
        } catch (error) {
          return `${String(error)} (${Notification.permission})`
        }
      }),
      'the browser refuses the constructor as Android does, with notifications allowed'
    ).toBe(`TypeError: ${MOBILE_REFUSAL} (granted)`)

    // A turn in chat A stops at an approval while the user is in another chat: the attention event.
    const victim = path.join(sandbox.root, `victim-notify-${nonce}`)
    fs.mkdirSync(victim)
    provider.script(U(1), [
      {
        text: [`${AI(1)} `, 'needs ', 'approval'],
        holdBeforeFinish: hold,
        toolCalls: [{ name: 'terminal', args: { command: `rm -rf ${shellQuote(victim)}` } }]
      },
      { text: [`${A(1)} `, 'deleted ', 'it'] }
    ])
    await send(page, `${U(1)} delete the dir`, 'Enter', ws)
    await provider.streamStarted(U(1))
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const sessionId = await currentSessionId(page)
    await page.evaluate(() => {
      window.location.hash = '#/'
    })
    await expect.poll(() => currentSessionId(page)).toBe('')
    // The Webapp holds OS notifications for 4 s after a socket opens, so that replayed prompts do
    // not alert (SEED_QUIET_MS, src/store/notify-baseline.ts). The approval is raised after that.
    await page.waitForTimeout(4_500)
    hold.open()

    const approvalTag = `hermes:approval:${sessionId}`

    // Within 15 s: headless Chromium closes a notification by itself after about 20 s, and a failure
    // should report what was on screen.
    await expect
      .poll(
        async () => ({
          approval: (await shownNotifications(page)).filter(shown => shown.tag === approvalTag),
          notify: await notifyOutcomes(page)
        }),
        { timeout: 15_000, message: "the worker's registration shows the approval's notification" }
      )
      .toEqual({
        approval: [
          {
            body: expect.stringContaining(victim),
            focusSessionId: sessionId,
            tag: approvalTag,
            title: expect.stringMatching(/^Approval needed/)
          }
        ],
        notify: ['approval: true']
      })
    expect(fs.existsSync(victim), 'the gated command waits for the answer').toBe(true)

    // The click, inside the real worker, on the notification it showed (see the header).
    const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'))
    expect(worker.url(), 'the notification worker is served at the Webapp root').toBe(
      `http://127.0.0.1:${port}/notification-worker.js`
    )
    await worker.evaluate(async tag => {
      const scope = globalThis as unknown as {
        dispatchEvent: (event: Event) => boolean
        NotificationEvent: new (type: string, init: { notification: Notification }) => Event
        registration: ServiceWorkerRegistration
      }

      const [notification] = await scope.registration.getNotifications({ tag })
      scope.dispatchEvent(new scope.NotificationEvent('notificationclick', { notification }))
    }, approvalTag)

    await expect.poll(() => currentSessionId(page), { message: 'the click opened the chat it named' }).toBe(sessionId)
    await expect(page.locator('[data-approval-run]').filter({ visible: true })).toHaveCount(1, { timeout: 60_000 })
    await expect
      .poll(async () => (await shownNotifications(page)).filter(shown => shown.tag === approvalTag))
      .toEqual([])

    // Settings' check, as a user runs it on the device.
    await page.evaluate(() => {
      window.location.hash = '#/settings?tab=notifications'
    })
    await page.getByRole('button', { name: 'Send test notification' }).click()
    await expect(page.getByText('Test sent.', { exact: false })).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText('This system does not support native notifications.')).toHaveCount(0)
    await expect
      .poll(async () => (await shownNotifications(page)).filter(shown => shown.tag === 'hermes:turnDone:'))
      .toEqual([
        { body: 'Notifications are working.', focusSessionId: undefined, tag: 'hermes:turnDone:', title: 'Hermes' }
      ])
    expect(await notifyOutcomes(page)).toEqual(['approval: true', 'turnDone: true'])
  } finally {
    hold.open()
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
})
