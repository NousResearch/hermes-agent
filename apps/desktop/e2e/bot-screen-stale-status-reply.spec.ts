import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Bot Screen portal vs. a slow `display.status` reply, in the real Electron app.
//
// The Screen hero (the portal above a bot's routines) fetches `display.status`
// once on mount; opening the Screen pane fetches it again. The renderer keeps a
// per-bot generation so a reply that a NEWER request overtook is dropped. Before
// the fix the portal's reply carried no request token, so `setScreenStatus`
// treated it as authoritative: it landed as truth AND bumped the generation,
// which dropped the pane's newer `running: false` — the hero kept saying
// "Live · bot in control" (running/pid/display of a screen that had stopped)
// and the pane attached to a dead display.
//
// The gateway's answer to `display.status` is the only thing scripted here:
// `WebSocket.prototype.send` holds the two requests and the test answers them
// through the real JSON-RPC client in the order the bug needs (portal's stale
// `running: true` first, pane's fresh `running: false` second). Everything else
// — roster, routines pane, hero, pane, store, listeners — is the shipped renderer.
//
// The orphan-row branch (a source-scoped row whose connection was deleted) is
// not reachable in this single-local-source rig: `window.hermesDesktop` is a
// non-writable contextBridge object, so the union roster cannot be seeded with
// a connection-less agent. That branch stays covered by
// `src/plugins/hermes-bots/screen-connection.test.ts`; the second test here
// drives the same shipped `display.lease` / `display.status` listeners with
// pushed events through the real client and asserts they keep updating without
// a page error.

type Page = MockBackendFixture['page']

interface ScreenProbe {
  held: Array<{ id: string | number }>
  log: string[]
}

let fixture: MockBackendFixture | null = null
const pageErrors: string[] = []

// BOT_SCREEN_SCREENSHOT_DIR=<dir> saves full-window captures at the key states.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_SCREEN_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
}

async function seedBot(hermesHome: string, mockUrl: string, name: string): Promise<void> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })
  } finally {
    await builder.close()
  }
}

const PROFILE_KEY = '/e2e/alpha/.hermes'

function snapshot(running: boolean) {
  return {
    profile: 'alpha',
    profile_key: PROFILE_KEY,
    supported: true,
    installed: true,
    missing: [],
    running,
    pid: running ? 4242 : null,
    display: running ? ':7' : null,
    socket: null,
    geometry: '1280x800',
    install_command: null,
    browser: null,
    lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, epoch: 1, reason: '' }
  }
}

/** Hold every outgoing `display.status` request; answer / push events through the real client. */
async function installScreenProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    if ((window as unknown as { __screenProbe?: unknown }).__screenProbe) {
      return
    }

    const held: Array<{ id: unknown; socket: WebSocket }> = []
    const log: string[] = []
    let lastSocket: WebSocket | null = null
    const send = WebSocket.prototype.send

    WebSocket.prototype.send = function (this: WebSocket, text: Parameters<WebSocket['send']>[0]) {
      lastSocket = this

      try {
        const frame = JSON.parse(String(text)) as { id?: unknown; method?: string }

        if (frame?.method === 'display.status') {
          held.push({ id: frame.id, socket: this })
          log.push(`held display.status #${String(frame.id)}`)

          return
        }
      } catch {
        // binary / non-JSON frames pass through untouched
      }

      return send.call(this, text)
    }

    const deliver = (socket: WebSocket, frame: Record<string, unknown>) =>
      socket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ jsonrpc: '2.0', ...frame }) }))

    ;(window as unknown as { __screenProbe: unknown }).__screenProbe = {
      held,
      log,
      answer(index: number, result: unknown) {
        const request = held[index]
        log.push(`answered #${String(request.id)} running=${String((result as { running?: boolean }).running)}`)
        deliver(request.socket, { id: request.id, result })
      },
      event(type: string, payload: unknown) {
        const socket = held[0]?.socket ?? lastSocket

        if (!socket) {
          throw new Error('no gateway socket has sent a frame yet')
        }

        log.push(`event ${type}`)
        deliver(socket, { method: 'event', params: { type, payload, session_id: '' } })
      }
    }
  })
}

const probe = (page: Page) => page.evaluate(() => {
  const state = (window as unknown as { __screenProbe: ScreenProbe }).__screenProbe

  return { held: state.held.length, log: [...state.log] }
})

const answer = (page: Page, index: number, result: unknown) =>
  page.evaluate(([i, r]) => (window as unknown as { __screenProbe: { answer: (i: number, r: unknown) => void } }).__screenProbe.answer(i, r), [index, result] as const)

const pushEvent = (page: Page, type: string, payload: unknown) =>
  page.evaluate(([t, p]) => (window as unknown as { __screenProbe: { event: (t: string, p: unknown) => void } }).__screenProbe.event(t, p), [type, payload] as const)

const hero = (page: Page) => page.locator('button[aria-label^="Screen:"]').first()

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-screen-stale')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url, 'alpha')

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  page.on('pageerror', error => pageErrors.push(String(error)))

  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a stale portal display.status reply does not roll back the newer stopped state', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await installScreenProbe(page)

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()

  const row = page.locator('[data-slot="bots-roster"] [data-roster-key="local::alpha"]')
  await expect(row).toBeVisible({ timeout: 30_000 })

  for (let attempt = 1; ; attempt += 1) {
    await row.click()

    try {
      await expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({ timeout: 45_000 })

      break
    } catch (error) {
      if (attempt >= 3) {
        throw error
      }
    }
  }

  await page
    .getByText(/Waking up/i)
    .first()
    .waitFor({ state: 'hidden', timeout: 90_000 })
    .catch(() => undefined)

  // Reveal the routines pane: the Screen hero mounts and fires the portal's one-shot fetch.
  if ((await hero(page).count()) === 0) {
    await page.getByRole('tab', { name: 'Scheduled jobs' }).first().click()
  }

  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Checking the screen…', { timeout: 15_000 })
  await expect.poll(async () => (await probe(page)).held, { timeout: 15_000 }).toBe(1)
  await capture(page, '1-hero-fetch-in-flight')

  // Open the pane: its own (newer) display.status request goes out while the portal's is still open.
  await hero(page).click()
  await expect.poll(async () => (await probe(page)).held, { timeout: 15_000 }).toBe(2)

  // The portal's older reply lands first and describes a screen that has since been
  // stopped; the pane's newer reply says so. Only the newer one may win.
  await answer(page, 0, snapshot(true))
  await answer(page, 1, snapshot(false))
  await page.waitForTimeout(1000)
  await capture(page, '2-after-stale-reply')

  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Screen is off', { timeout: 10_000 })
  await expect(page.getByText('Screen is off', { exact: true }).filter({ visible: true }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Start screen' })).toBeVisible()

  // Belt and braces: it stays stopped — nothing later "corrects" back to live.
  await page.waitForTimeout(1500)
  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Screen is off')
  expect(pageErrors).toEqual([])
})

test('pushed display.status / display.lease events keep updating the shipped listeners without a page error', async () => {
  test.setTimeout(120_000)
  const page = fixture!.page

  await installScreenProbe(page)
  await expect(hero(page)).toBeVisible({ timeout: 15_000 })

  // A start made outside this window is pushed as a token-less status: authoritative.
  await pushEvent(page, 'display.status', snapshot(true))
  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Live · bot in control', { timeout: 10_000 })

  // An event for another screen (different profile_key) is ignored by every listener.
  await pushEvent(page, 'display.lease', {
    profile_key: '/e2e/other/.hermes',
    lease: { holder: 'human', viewer_id: null, viewer_hash: 'someone-else', since: 2, epoch: 2, reason: '' }
  })
  await page.waitForTimeout(500)
  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Live · bot in control')

  // A takeover on this screen by another viewer repaints the hero and the pane.
  await pushEvent(page, 'display.lease', {
    profile_key: PROFILE_KEY,
    lease: { holder: 'human', viewer_id: null, viewer_hash: 'someone-else', since: 2, epoch: 2, reason: '' }
  })
  await expect(hero(page)).toHaveAttribute('aria-label', 'Screen: Live · another viewer in control', { timeout: 10_000 })
  await capture(page, '3-lease-event-applied')

  expect(pageErrors).toEqual([])
})
