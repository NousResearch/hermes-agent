/**
 * E2E: a bot on a REMOTE gateway appears in the normal bot list, attributed to
 * that gateway (`gateway-scoped-bots.md` §6, §8.1, §11.1).
 *
 * "This device" is the Electron-managed local backend. `cloud-123` is a saved
 * remote entry carrying the `hermes cloud` shape: `kind: remote`,
 * `authMode: token`, the cloud instance `name`, and a loopback Desktop URL. Its
 * backend is a fake `hermes serve` (HTTP + WS) on a fixed port, so the test can
 * stop and restart exactly that gateway:
 *
 *   healthy   → row visible with the gateway badge, under its own section
 *   stopped   → only that section flips to Unavailable; the row is retained and
 *               local rows are untouched
 *   restarted → the section recovers on the next poll, no app restart
 *
 * The roster reply is asserted too (`sources[].cloud`, `agents[].gatewayName`).
 * Screenshots land in HERMES_E2E_OUT when set. Needs `pnpm run build` (dist/).
 */

import * as fs from 'node:fs'
import * as http from 'node:http'
import { createRequire } from 'node:module'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { insideHermesProfileHome } from './sandbox-root'
import { type ElectronApplication, expect, type Page, test } from './test'

const CLOUD_ID = 'cloud-123'
const CLOUD_NAME = 'cloud-123'
const CLOUD_TOKEN = 'e2e-cloud-123-peer-key'
const CLOUD_INSTALL_ID = 'cloud-gw-install-123'

const outPath = (file: string): string =>
  process.env.HERMES_E2E_OUT ? path.join(process.env.HERMES_E2E_OUT, file) : test.info().outputPath(file)

interface FakeSocket {
  send: (data: string) => void
  on: (event: 'message', listener: (raw: unknown) => void) => void
  terminate: () => void
}

interface FakeSocketServer {
  clients: Set<FakeSocket>
  on: (event: 'connection', listener: (ws: FakeSocket) => void) => void
  close: () => void
}

const { WebSocketServer } = createRequire(import.meta.url)('ws') as {
  WebSocketServer: new (options: { server: http.Server }) => FakeSocketServer
}

interface FakeGateway {
  url: string
  port: number
  close: () => Promise<void>
}

/** A `hermes serve`-shaped backend for ONE cloud bot: token-gated
 *  `/api/profiles`, a public `/api/status` carrying `install_id`, and a WS that
 *  answers the boot RPCs so the source looks like a configured gateway. */
async function startCloudGateway(port: number): Promise<FakeGateway> {
  const server = http.createServer((req, res) => {
    req.resume()
    const pathname = new URL(req.url ?? '/', 'http://x').pathname
    res.setHeader('Content-Type', 'application/json')

    const presented = String(req.headers['x-hermes-session-token'] || req.headers.authorization || '').replace(
      /^Bearer\s+/i,
      ''
    )

    if (pathname === '/api/status') {
      // Public on loopback, exactly like the real backend (`/api/status` is not
      // the auth proof — `/api/profiles` is, §10 trap 2).
      return res.end(
        JSON.stringify({ auth_required: true, install_id: CLOUD_INSTALL_ID, status: 'ok', version: '0.23.0' })
      )
    }

    if (pathname === '/api/profiles') {
      if (presented !== CLOUD_TOKEN) {
        res.statusCode = 401

        return res.end(JSON.stringify({ error: 'unauthorized' }))
      }

      return res.end(
        JSON.stringify({
          profiles: [
            {
              active: true,
              bot_title: 'Cloud 123',
              description: 'Bot running on the cloud-123 gateway',
              display_name: 'Cloud 123',
              distribution_name: null,
              distribution_source: null,
              distribution_version: null,
              has_alias: false,
              has_env: true,
              is_default: true,
              model: 'mock-model',
              name: 'default',
              path: '/sandbox/profiles/default',
              provider: 'custom',
              gateway_running: true,
              skill_count: 3
            }
          ],
          active: 'default'
        })
      )
    }

    if (pathname === '/api/profiles/active') {
      return res.end(JSON.stringify({ name: 'default', profile: 'default' }))
    }

    if (pathname === '/api/sessions') {
      return res.end(JSON.stringify({ sessions: [], total: 0 }))
    }

    if (pathname.startsWith('/api/config')) {
      return res.end(JSON.stringify({}))
    }

    res.statusCode = 404
    res.end(JSON.stringify({ error: 'not_found' }))
  })

  const wss = new WebSocketServer({ server })

  wss.on('connection', ws => {
    ws.send(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'gateway.ready', payload: { change_events: true, heartbeat: false, replay_epoch: CLOUD_ID } }
      })
    )
    ws.on('message', raw => {
      let msg: { id?: unknown; method?: string } = {}

      try {
        msg = JSON.parse(String(raw))
      } catch {
        return
      }

      if (msg.id !== undefined) {
        const result =
          msg.method === 'setup.status'
            ? { provider_configured: true, ready: true }
            : msg.method === 'setup.runtime_check'
              ? { ok: true, provider: 'custom', model: 'mock-model' }
              : msg.method === 'onboarding.state'
                ? { active: false, completed: true }
                : {}

        ws.send(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }))
      }
    })
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(port, '127.0.0.1', () => resolve())
  })

  return {
    port,
    url: `http://127.0.0.1:${port}`,
    close: async () => {
      for (const client of wss.clients) {
        client.terminate()
      }

      wss.close()
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }
}

function cloudConnectionEntry(cloudUrl: string) {
  return {
    id: CLOUD_ID,
    kind: 'remote',
    label: CLOUD_NAME,
    name: CLOUD_NAME,
    url: cloudUrl,
    authMode: 'token',
    token: { encoding: 'plain', value: CLOUD_TOKEN }
  }
}

function writeConnectionsRegistry(sandboxRoot: string, connections: unknown[]): void {
  const userDataDir = path.join(sandboxRoot, 'electron-user-data')

  fs.writeFileSync(
    path.join(userDataDir, 'connections.json'),
    JSON.stringify(
      {
        version: 2,
        primary: 'local',
        launchMode: 'primary',
        lastUsed: 'local',
        connections
      },
      null,
      2
    ),
    { encoding: 'utf8', mode: 0o600 }
  )
}

/** Register the cloud gateway the way `hermes cloud create` does — by appending
 *  its entry to the machine's shared store (`<HERMES_HOME>/gateway-connections.json`,
 *  the file the Desktop and the Webapp both write and the watcher re-reads every
 *  1.5 s), NOT to the userData registry the boot import consumed. This is what
 *  proves acceptance item 1: an entry appearing while the app runs shows up in
 *  the roster on the next poll with no restart. */
function appendCloudConnectionToStore(hermesHome: string, cloudUrl: string): void {
  const storePath = path.join(hermesHome, 'gateway-connections.json')
  const doc = JSON.parse(fs.readFileSync(storePath, 'utf8'))
  const connections = Array.isArray(doc.connections) ? doc.connections : []

  if (!connections.some((entry: { id?: string }) => entry?.id === CLOUD_ID)) {
    connections.push(cloudConnectionEntry(cloudUrl))
  }

  fs.writeFileSync(
    storePath,
    JSON.stringify({ ...doc, connections, revision: Number(doc.revision || 0) + 1 }, null, 2),
    { encoding: 'utf8', mode: 0o600 }
  )
}

async function openBots(page: Page): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible({ timeout: 60_000 })
}

const cloudRow = (page: Page) => page.locator(`[data-slot="bots-roster"] [data-roster-key="${CLOUD_ID}::default"]`)

test.describe('remote-gateway bot in the roster', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: ReturnType<typeof createSandbox>
  let cloud: FakeGateway
  let app: ElectronApplication
  let page: Page
  const cloudPort = 18764

  test.beforeAll(async () => {
    test.setTimeout(300_000)
    mock = await startMockServer()
    sandbox = createSandbox('cloud-roster')

    // A sandbox under `<hermes home>/profiles/<bot>/…` reads as that bot's profile home, and the
    // gateway then parks the sandbox profile ("Gateway inaccessible (profile_parked). only one
    // profile exists (nothing to multiplex)") instead of booting it — the app never reaches ready.
    // createSandbox avoids that base; fail loudly here rather than burning the boot timeout.
    expect(insideHermesProfileHome(sandbox.root), 'sandbox root must not be inside a profile home').toBe(false)

    // `multiplex_profiles` is what keeps an isolated sandbox backend from
    // parking its only profile ("Gateway inaccessible (profile_parked)"); the
    // version stamp keeps the boot migration from treating a test-written
    // config as pre-history.
    writeMockProviderConfig(
      sandbox.hermesHome,
      mock.url,
      undefined,
      '_config_version: 53\ngateway:\n  multiplex_profiles: true\n'
    )
    writeEnvFile(sandbox.hermesHome, undefined, mock.url)

    // The host gateway is multiplex-only and refuses a single-profile install
    // ("nothing to multiplex"). A second profile makes the sandbox a normal
    // two-bot "This device" source — the local rows this test must not disturb.
    const scribeHome = path.join(sandbox.hermesHome, 'profiles', 'scribe')

    fs.mkdirSync(scribeHome, { recursive: true })
    writeMockProviderConfig(scribeHome, mock.url, undefined, '_config_version: 53\n')
    writeEnvFile(scribeHome, undefined, mock.url)

    // The sandbox's HERMES_HOME is pinned by buildAppEnv (this is the home `hermes gateway ensure`
    // resolves). The copy under $HOME/.hermes only covers paths that resolve off HOME instead; it is
    // not the fix for the parked-profile failure (see sandbox-root.ts for that).
    const homeAnchored = path.join(sandbox.root, '.hermes')

    fs.mkdirSync(homeAnchored, { recursive: true })
    writeMockProviderConfig(homeAnchored, mock.url, undefined, '_config_version: 53\n')
    writeEnvFile(homeAnchored, undefined, mock.url)
    const homeAnchoredScribe = path.join(homeAnchored, 'profiles', 'scribe')

    fs.mkdirSync(homeAnchoredScribe, { recursive: true })
    writeMockProviderConfig(homeAnchoredScribe, mock.url, undefined, '_config_version: 53\n')
    writeEnvFile(homeAnchoredScribe, undefined, mock.url)
    cloud = await startCloudGateway(cloudPort)
    // Boot the app with ONLY the built-in local source: the cloud entry is
    // registered later, while the app runs, in the first test.
    writeConnectionsRegistry(sandbox.root, [{ id: 'local', kind: 'local', label: 'This device' }])
    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({
      timeout: 120_000
    })
    await page.waitForTimeout(2_000)
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await cloud?.close().catch(() => undefined)
    await mock?.close()

    // Debug hook: `HERMES_E2E_KEEP=1` leaves the sandbox (and its desktop.log).
    if (!process.env.HERMES_E2E_KEEP) {
      sandbox?.cleanup()
    }
  })

  test('registered while running: the cloud bot joins the list on the next poll', async () => {
    test.setTimeout(180_000)
    await openBots(page)

    // Before: only the built-in local source, no cloud row anywhere.
    await expect(cloudRow(page)).toHaveCount(0)
    await expect(page.locator('[data-slot="bots-roster"]')).not.toContainText(/cloud-123/i)

    const before = await page.evaluate(() => (window as any).hermesDesktop.getAgentRoster())

    expect(before.sources.some((entry: any) => entry.connectionId === CLOUD_ID)).toBe(false)

    // `hermes cloud create` appends the entry to the shared store; the Desktop
    // re-reads it on its own ~5 s roster poll. Acceptance: visible in the SAME
    // list, with its gateway identity, without restarting the app.
    appendCloudConnectionToStore(sandbox.hermesHome, cloud.url)

    await expect(cloudRow(page)).toBeVisible({ timeout: 30_000 })
    await expect(cloudRow(page).locator(`[data-gateway-badge="${CLOUD_NAME}"]`)).toBeVisible()
    await expect(page.locator('[data-slot="bots-roster"]')).toContainText(/cloud-123/i)
  })

  test('healthy gateway: the row and its gateway identity are in the same list', async () => {
    await openBots(page)

    // The IPC reply carries the gateway identity, and the roster row is a
    // normal row (same component) tagged with the gateway name.
    const reply = await page.evaluate(() => (window as any).hermesDesktop.getAgentRoster())
    const source = reply.sources.find((entry: any) => entry.connectionId === CLOUD_ID)
    const agent = reply.agents.find((entry: any) => entry.connectionId === CLOUD_ID)

    fs.writeFileSync(outPath('roster-reply.json'), JSON.stringify(reply, null, 2))

    expect(source).toMatchObject({
      connectionId: CLOUD_ID,
      kind: 'remote',
      label: CLOUD_NAME,
      name: CLOUD_NAME,
      reachable: true,
      url: cloud.url,
      cloud: { bot: CLOUD_NAME, gateway_id: CLOUD_INSTALL_ID }
    })
    expect(agent?.gatewayName, 'the cloud row carries its gateway name').toBe(CLOUD_NAME)
    expect(reply.agents.find((entry: any) => entry.connectionId === 'local')?.gatewayName).toBeUndefined()

    const row = cloudRow(page)
    await expect(row).toBeVisible({ timeout: 60_000 })
    await expect(row.locator(`[data-gateway-badge="${CLOUD_NAME}"]`)).toBeVisible()
    await expect(row).toContainText('Cloud 123')

    // Same list: the local row is beside it, and the primary section leads
    // (D7 — "cloud-123" sorts before "This device" alphabetically).
    // innerText is RENDERED text: section headings carry `uppercase`, so compare case-insensitively.
    const text = (await page.locator('[data-slot="bots-roster"]').innerText()).toLowerCase()
    expect(text).toContain('this device')
    expect(text).toContain('cloud-123')
    expect(text.indexOf('this device'), 'primary section first').toBeLessThan(text.indexOf('cloud-123'))

    // …and the DOM agrees: one section per gateway, primary first.
    const sectionOrder = await page
      .locator('[data-slot="bots-roster"] [data-roster-section]')
      .evaluateAll(elements => elements.map(element => element.getAttribute('data-roster-section')))

    expect(sectionOrder).toEqual(['gateway:local', 'gateway:cloud-123'])

    await page.screenshot({ path: outPath('1-healthy-both-gateways.png') })
  })

  test('gateway stopped: only its section flips to Unavailable, the row stays', async () => {
    test.setTimeout(180_000)
    await cloud.close()

    // One ~5 s roster poll (plus enumeration headroom) is the acceptance window. The status label is
    // rendered sr-only behind the amber disconnect glyph, so assert on the section heading's text
    // content (its `sr-only` label) per section — NOT on globe text, which would also pass if the
    // local section went unavailable.
    const roster = page.locator('[data-slot="bots-roster"]')
    const cloudHeading = page.locator('[data-roster-section="gateway:cloud-123"]')
    const localHeading = page.locator('[data-roster-section="gateway:local"]')

    await expect(cloudHeading).toContainText(/unavailable/i, { timeout: 60_000 })
    await expect(localHeading).not.toContainText(/unavailable/i)
    // The row stays, keeps its gateway identity, and every local row is untouched.
    await expect(cloudRow(page)).toBeVisible()
    await expect(cloudRow(page).locator(`[data-gateway-badge="${CLOUD_NAME}"]`)).toBeVisible()
    await expect(page.locator(`[data-slot="bots-roster"] [data-roster-key="local::default"]`)).toBeVisible()
    await expect(page.locator(`[data-slot="bots-roster"] [data-roster-key="local::scribe"]`)).toBeVisible()
    await expect(roster).toBeVisible()
    await page.screenshot({ path: outPath('2-cloud-gateway-unavailable.png') })
  })

  test('gateway restored: the section recovers on the next poll, no app restart', async () => {
    test.setTimeout(180_000)
    // Same port, same URL — exactly what restarting the supervisor does.
    await new Promise(resolve => setTimeout(resolve, 1_500))
    cloud = await startCloudGateway(cloudPort)

    const cloudHeading = page.locator('[data-roster-section="gateway:cloud-123"]')

    await expect(cloudHeading).not.toContainText(/unavailable/i, { timeout: 90_000 })
    await expect(page.locator('[data-slot="bots-roster"]')).not.toContainText(/unavailable/i)
    await expect(cloudRow(page)).toBeVisible()
    await expect(cloudRow(page).locator(`[data-gateway-badge="${CLOUD_NAME}"]`)).toBeVisible()
    await page.screenshot({ path: outPath('3-recovered.png') })
  })
})
