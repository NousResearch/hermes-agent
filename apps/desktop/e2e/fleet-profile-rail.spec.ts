/**
 * E2E: the fleet profile rail with two registered gateways.
 *
 * "This device" is the Electron-managed local backend (mock inference). The
 * second gateway, "Homelab", is a REAL second `hermes serve` this spec spawns
 * with its own HERMES_HOME, profiles and session token, registered in the v2
 * connections.json as a remote URL connection. A click on an at-rest square
 * therefore performs the same dial → commit → re-home the statusbar switcher
 * does, against a real backend — not a stub.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists, and the
 * repo's Python venv (`.venv`) must exist for both backends.
 */

import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { type ElectronApplication, expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

const REMOTE_LABEL = 'Homelab'
const REMOTE_ID = 'homelab'
const REMOTE_TOKEN = 'e2e-fleet-homelab-token'

interface RemoteGateway {
  url: string
  home: string
  close: () => Promise<void>
}

function findHermesBinary(): string {
  const venv = path.join(REPO_ROOT, '.venv', 'bin', 'hermes')

  if (fs.existsSync(venv)) {
    return venv
  }

  const result = spawnSync('which', ['hermes'], { encoding: 'utf8' })

  if (result.status === 0 && result.stdout.trim()) {
    return result.stdout.trim()
  }

  throw new Error('hermes binary not found: create the repo venv (uv sync) or put hermes on PATH')
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

/** Seed `<home>/profiles/<name>/` so the backend's /api/profiles lists it. */
function seedProfiles(home: string, names: string[]): void {
  for (const name of names) {
    const dir = path.join(home, 'profiles', name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'config.yaml'), '', 'utf8')
  }
}

/**
 * Spawn a second, fully real `hermes serve` as the remote gateway. Its
 * session token is pinned through HERMES_DASHBOARD_SESSION_TOKEN so the
 * registry entry can carry a plaintext token envelope.
 */
async function startRemoteGateway(root: string, mockUrl: string, profiles: string[]): Promise<RemoteGateway> {
  const home = path.join(root, 'homelab-home')
  fs.mkdirSync(home, { recursive: true })
  writeMockProviderConfig(home, mockUrl)
  writeEnvFile(home)
  seedProfiles(home, profiles)

  const port = await freePort()
  const url = `http://127.0.0.1:${port}`

  const child: ChildProcess = spawn(
    findHermesBinary(),
    ['serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'],
    {
      cwd: REPO_ROOT,
      detached: true,
      env: {
        ...process.env,
        HERMES_HOME: home,
        HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )

  let log = ''
  child.stdout?.on('data', (chunk: Buffer) => {
    log += chunk.toString()
  })
  child.stderr?.on('data', (chunk: Buffer) => {
    log += chunk.toString()
  })

  const deadline = Date.now() + 90_000

  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`remote hermes serve exited early (${child.exitCode}):\n${log}`)
    }

    try {
      const response = await fetch(`${url}/api/status`, {
        headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN },
      })

      if (response.ok) {
        break
      }
    } catch {
      // not up yet
    }

    await new Promise(resolve => setTimeout(resolve, 500))
  }

  if (Date.now() >= deadline) {
    throw new Error(`remote hermes serve never became ready:\n${log}`)
  }

  return {
    url,
    home,
    close: async () => {
      if (child.pid && child.exitCode === null) {
        try {
          process.kill(-child.pid, 'SIGTERM')
        } catch {
          child.kill('SIGTERM')
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500))
    },
  }
}

function writeConnectionsRegistry(sandbox: Sandbox, remoteUrl: string): void {
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connections.json'),
    JSON.stringify(
      {
        version: 2,
        primary: 'local',
        launchMode: 'primary',
        lastUsed: 'local',
        connections: [
          { id: 'local', kind: 'local', label: 'This device' },
          {
            id: REMOTE_ID,
            kind: 'remote',
            label: REMOTE_LABEL,
            url: remoteUrl,
            authMode: 'token',
            token: { encoding: 'plain', value: REMOTE_TOKEN },
          },
        ],
      },
      null,
      2,
    ),
    { encoding: 'utf8', mode: 0o600 },
  )
}

// FLEET_RAIL_SCREENSHOT_DIR=<dir> saves full-window captures at the key
// states — handy for design review; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.FLEET_RAIL_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
}

const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')
const gatewayGroup = (page: Page, id: string) => rail(page).locator(`[data-slot="profile-rail-gateway"][data-connection-id="${id}"]`)
const activeGatewayLabel = (page: Page) => page.getByRole('button', { name: /^Registered gateways: / })

async function groupOrder(page: Page): Promise<Array<[string, boolean]>> {
  return rail(page).locator('[data-slot="profile-rail-gateway"]').evaluateAll(nodes =>
    nodes.map(node => [node.getAttribute('data-connection-id') ?? '', node.getAttribute('data-active') === 'true'] as [string, boolean]),
  )
}

test.describe('fleet profile rail — two registered gateways', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let remote: RemoteGateway
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    mock = await startMockServer()
    sandbox = createSandbox('fleet')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    // A named profile on This device too, so the active group has a square
    // beside its home pill. "research" exists on BOTH gateways on purpose: the
    // rail must keep the two apart by gateway, never by name alone.
    seedProfiles(sandbox.hermesHome, ['research'])

    remote = await startRemoteGateway(sandbox.root, mock.url, ['inbox', 'research'])
    writeConnectionsRegistry(sandbox, remote.url)

    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    // Let boot settle fully (the gateway health item reports "ready" once the
    // primary socket is open) so the boot-time launch-mode restore has run
    // before any click — the rail must then hold whatever the user picks.
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
    await page.waitForTimeout(2_000)
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await remote?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('lays both gateways on one strip, active gateway in its registry slot', async () => {
    // The statusbar readout names the gateway the workspace is on.
    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', 'Registered gateways: This device', { timeout: 60_000 })

    // The remote gateway's group appears once the roster has enumerated it.
    const homelab = gatewayGroup(page, REMOTE_ID)
    await expect(homelab).toBeVisible({ timeout: 60_000 })
    await expect(homelab.getByRole('button', { name: `default · ${REMOTE_LABEL}` })).toBeVisible()
    await expect(homelab.getByRole('button', { name: `inbox · ${REMOTE_LABEL}` })).toBeVisible()
    await expect(homelab.getByRole('button', { name: `research · ${REMOTE_LABEL}` })).toBeVisible()
    await expect(homelab).toHaveAttribute('data-reachable', 'true')

    // Its marker carries the remote (network) glyph.
    await expect(
      rail(page).locator(`[data-slot="profile-rail-divider"][data-connection-id="${REMOTE_ID}"] [data-connection-kind="remote"]`),
    ).toBeVisible()

    // This device is the active group: its squares are unqualified, as before.
    const local = gatewayGroup(page, 'local')
    await expect(local).toHaveAttribute('data-active', 'true')
    await expect(local.getByRole('button', { name: 'research', exact: true })).toBeVisible()

    // Registry order: This device first, Homelab second.
    expect(await groupOrder(page)).toEqual([
      ['local', true],
      [REMOTE_ID, false],
    ])

    // Fleet pill replaces the default↔all toggle; the single-gateway plug is gone.
    await expect(rail(page).getByRole('button', { name: 'All profiles on this gateway' })).toBeVisible()
    await expect(rail(page).getByRole('button', { name: 'Manage gateways…' })).toHaveCount(0)

    await gatewayGroup(page, REMOTE_ID).getByRole('button', { name: `inbox · ${REMOTE_LABEL}` }).hover()
    await capture(page, '1-on-this-device-hover-inbox-homelab')
  })

  test('clicking an at-rest square re-homes onto that exact gateway and profile', async () => {
    test.setTimeout(180_000)
    await gatewayGroup(page, REMOTE_ID).getByRole('button', { name: `inbox · ${REMOTE_LABEL}` }).click()

    // The workspace follows the agent: statusbar readout flips to Homelab…
    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', `Registered gateways: ${REMOTE_LABEL}`, { timeout: 120_000 })

    // …Homelab's group is now the active one, on the clicked profile…
    const homelab = gatewayGroup(page, REMOTE_ID)
    await expect(homelab).toHaveAttribute('data-active', 'true', { timeout: 30_000 })
    await expect(homelab.getByRole('button', { name: 'inbox', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 30_000 })

    // …This device is at rest with qualified squares…
    const local = gatewayGroup(page, 'local')
    await expect(local).toHaveAttribute('data-active', 'false')
    await expect(local.getByRole('button', { name: 'research · This device' })).toBeVisible()

    // …and nothing moved: the order is still This device, then Homelab.
    expect(await groupOrder(page)).toEqual([
      ['local', false],
      [REMOTE_ID, true],
    ])

    await capture(page, '2-re-homed-on-homelab-inbox')
  })

  test('an at-rest square offers gateway-scoped actions, never the legacy remote override', async () => {
    const square = gatewayGroup(page, 'local').getByRole('button', { name: 'research · This device' })
    await square.click({ button: 'right' })

    const menu = page.getByRole('menu', { name: 'Actions' })
    await expect(menu).toBeVisible()
    await expect(menu.getByRole('menuitem', { name: 'Switch to research on This device' })).toBeVisible()
    await expect(menu.getByRole('menuitem', { name: 'Rename…' })).toBeVisible()
    await expect(menu.getByRole('menuitem', { name: 'Edit SOUL.md…' })).toBeVisible()
    await expect(menu.getByRole('menuitem', { name: 'Delete' })).toBeVisible()
    await expect(menu.getByRole('menuitem', { name: 'Connect to a remote host…' })).toHaveCount(0)

    await capture(page, '3-at-rest-square-context-menu')
    await page.keyboard.press('Escape')
    await expect(menu).toBeHidden()
  })

  test('editing SOUL.md on an at-rest square reads the owning gateway, not the foreground one', async () => {
    const square = gatewayGroup(page, 'local').getByRole('button', { name: 'research · This device' })
    await square.click({ button: 'right' })
    await page.getByRole('menu', { name: 'Actions' }).getByRole('menuitem', { name: 'Edit SOUL.md…' }).click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText('research · This device · SOUL.md')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(dialog).toBeHidden()
  })

  test('switching back lands on the clicked profile of This device and keeps the order', async () => {
    test.setTimeout(180_000)
    await gatewayGroup(page, 'local').getByRole('button', { name: 'research · This device' }).click()

    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', 'Registered gateways: This device', { timeout: 120_000 })
    const local = gatewayGroup(page, 'local')
    await expect(local).toHaveAttribute('data-active', 'true', { timeout: 30_000 })
    await expect(local.getByRole('button', { name: 'research', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 30_000 })
    await expect(gatewayGroup(page, REMOTE_ID).getByRole('button', { name: `inbox · ${REMOTE_LABEL}` })).toBeVisible()

    expect(await groupOrder(page)).toEqual([
      ['local', true],
      [REMOTE_ID, false],
    ])
  })
})
