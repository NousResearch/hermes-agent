/**
 * E2E: the fleet profile rail when the registry PRIMARY is a remote gateway.
 *
 * The workspace boots straight onto "Homelab" (a real second `hermes serve`
 * this spec spawns) and Electron deliberately does not start a local backend
 * — "This device" is connect-on-demand. The rail must still show This device
 * at rest with its real squares (inventoried from the profiles directory),
 * without the amber "unreachable" mark; the first click dials it; and after
 * switching back to Homelab the group must repaint from a fresh roster, not
 * the boot snapshot.
 *
 * Prerequisite: `npm run build` (dist/) and a Python venv for both backends.
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
import { startMockServer } from './mock-server'
import { type ElectronApplication, expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

const REMOTE_LABEL = 'Homelab'
const REMOTE_ID = 'homelab'
const REMOTE_TOKEN = 'e2e-fleet-remote-primary-token'
const LOCAL_NAMED_PROFILE = 'research'

interface RemoteGateway {
  url: string
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

function seedProfiles(home: string, names: string[]): void {
  for (const name of names) {
    const dir = path.join(home, 'profiles', name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'config.yaml'), '', 'utf8')
  }
}

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
      env: { ...process.env, HERMES_HOME: home, HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN },
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
      const response = await fetch(`${url}/api/status`, { headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN } })

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

/** Registry whose PRIMARY is the remote gateway — the local entry is on demand. */
function writeConnectionsRegistry(sandbox: Sandbox, remoteUrl: string): void {
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connections.json'),
    JSON.stringify(
      {
        version: 2,
        primary: REMOTE_ID,
        launchMode: 'primary',
        lastUsed: REMOTE_ID,
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
// states — for review; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.FLEET_RAIL_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })

  // A close-up of the strip + statusbar, where the change is actually visible.
  const box = await rail(page).boundingBox()

  if (box) {
    const viewport = page.viewportSize() ?? { height: box.y + box.height + 40, width: box.x + box.width }
    const x = Math.max(0, box.x - 8)
    const y = Math.max(0, box.y - 12)
    await page.screenshot({
      clip: { height: Math.min(viewport.height - y, box.height + 48), width: Math.min(viewport.width - x, 420), x, y },
      path: path.join(dir, `${name}-rail.png`),
    })
  }
}

const botsTab = (page: Page) => page.getByRole('button', { name: /^bots$/i }).or(page.getByText(/^bots$/i)).first()
const sessionsTab = (page: Page) => page.getByRole('button', { name: /^sessions$/i }).or(page.getByText(/^sessions$/i)).first()

const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')
const gatewayGroup = (page: Page, id: string) => rail(page).locator(`[data-slot="profile-rail-gateway"][data-connection-id="${id}"]`)

const unreachableMark = (page: Page, id: string) =>
  rail(page).locator(`[data-slot="profile-rail-divider"][data-connection-id="${id}"] [data-slot="profile-rail-unreachable"]`)

const activeGatewayLabel = (page: Page) => page.getByRole('button', { name: /^Registered gateways: / })

/** Local `hermes serve` children spawned by this Electron app (by HERMES_HOME). */
function localBackendCount(hermesHome: string): number {
  const result = spawnSync('pgrep', ['-f', `hermes_cli.main --profile .* serve`], { encoding: 'utf8' })

  if (result.status !== 0) {
    return 0
  }

  let count = 0

  for (const pid of result.stdout.split('\n').map(line => line.trim()).filter(Boolean)) {
    try {
      const environ = fs.readFileSync(`/proc/${pid}/environ`, 'utf8')

      if (environ.includes(`HERMES_HOME=${hermesHome}\0`)) {
        count += 1
      }
    } catch {
      // process gone or unreadable
    }
  }

  return count
}

test.describe('fleet profile rail — remote primary, This device on demand', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let remote: RemoteGateway
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    mock = await startMockServer()
    sandbox = createSandbox('fleet-remote-primary')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    // A named profile on This device: it must appear on the strip from the
    // profiles directory alone, before any local backend has ever run.
    seedProfiles(sandbox.hermesHome, [LOCAL_NAMED_PROFILE])

    remote = await startRemoteGateway(sandbox.root, mock.url, ['inbox'])
    writeConnectionsRegistry(sandbox, remote.url)

    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
    await page.waitForTimeout(2_000)
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await remote?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('boots onto the remote primary with This device at rest — real squares, no unreachable mark, no local backend', async () => {
    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', `Registered gateways: ${REMOTE_LABEL}`, { timeout: 60_000 })
    await expect(gatewayGroup(page, 'local')).toBeVisible({ timeout: 60_000 })
    await capture(page, '1-boot-on-remote-primary-this-device-at-rest')
    await expect(gatewayGroup(page, REMOTE_ID)).toHaveAttribute('data-active', 'true')

    // This device: at rest, reachable (connect-on-demand is not a failure),
    // and its named square is already there — read from disk, not a backend.
    const local = gatewayGroup(page, 'local')
    await expect(local).toBeVisible({ timeout: 60_000 })
    await expect(local).toHaveAttribute('data-active', 'false')
    await expect(local.getByRole('button', { name: `${LOCAL_NAMED_PROFILE} · This device` })).toBeVisible({ timeout: 60_000 })
    await expect(local).toHaveAttribute('data-reachable', 'true')
    await expect(unreachableMark(page, 'local')).toHaveCount(0)

    // …and Electron has NOT started a local backend to learn that.
    if (localBackendCount(sandbox.hermesHome) !== 0) {
      const ps = spawnSync('ps', ['-eo', 'pid,ppid,etimes,args'], { encoding: 'utf8' }).stdout
        .split('\n')
        .filter(line => line.includes('hermes_cli.main'))
        .join('\n')

      let log = ''

      try {
        log = fs.readFileSync(path.join(sandbox.hermesHome, 'logs', 'desktop.log'), 'utf8')
      } catch {
        log = '(no desktop.log)'
      }

      console.log(`--- hermes processes ---\n${ps}\n--- desktop.log ---\n${log.slice(-12_000)}`)
    }

    expect(localBackendCount(sandbox.hermesHome)).toBe(0)
  })

  test('Bot Mode lists This device at boot, before any local backend has run', async () => {
    await botsTab(page).click()
    await page.waitForTimeout(3_000)
    await capture(page, '1b-bot-mode-lists-this-device-at-boot')
    // Bot Mode groups by gateway: the local group carries its home bot and
    // the named `research` profile — rows that only exist because the roster
    // seeded them from the profiles directory.
    await expect(page.getByText(/^this device$/i).first()).toBeVisible({ timeout: 60_000 })
    await expect(page.getByText('Research', { exact: true })).toBeVisible({ timeout: 60_000 })
    await expect(page.getByText('Inbox', { exact: true })).toBeVisible()
    expect(localBackendCount(sandbox.hermesHome)).toBe(0)
    await sessionsTab(page).click()
    await expect(gatewayGroup(page, 'local')).toBeVisible()
  })

  test('the first click on a This-device square dials it and re-homes there', async () => {
    test.setTimeout(180_000)
    await gatewayGroup(page, 'local').getByRole('button', { name: `${LOCAL_NAMED_PROFILE} · This device` }).click()

    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', 'Registered gateways: This device', { timeout: 120_000 })
    const local = gatewayGroup(page, 'local')
    await expect(local).toHaveAttribute('data-active', 'true', { timeout: 30_000 })
    await capture(page, '2-after-click-re-homed-on-this-device')
    await expect(local.getByRole('button', { name: LOCAL_NAMED_PROFILE, exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 30_000 })
    await expect(gatewayGroup(page, REMOTE_ID)).toHaveAttribute('data-active', 'false')
    await expect(gatewayGroup(page, REMOTE_ID)).toHaveAttribute('data-reachable', 'true')

    // Now a local backend exists — the click is what started it.
    expect(localBackendCount(sandbox.hermesHome)).toBeGreaterThan(0)
  })

  test('switching back to the remote leaves This device at rest and reachable — not the stale boot snapshot', async () => {
    test.setTimeout(180_000)
    await gatewayGroup(page, REMOTE_ID).getByRole('button', { name: `inbox · ${REMOTE_LABEL}` }).click()

    await expect(activeGatewayLabel(page)).toHaveAttribute('aria-label', `Registered gateways: ${REMOTE_LABEL}`, { timeout: 120_000 })
    await expect(gatewayGroup(page, REMOTE_ID)).toHaveAttribute('data-active', 'true', { timeout: 30_000 })

    const local = gatewayGroup(page, 'local')
    await expect(local).toHaveAttribute('data-active', 'false', { timeout: 30_000 })
    await page.waitForTimeout(1_500)
    await capture(page, '3-switched-back-this-device-still-reachable')
    await expect(local.getByRole('button', { name: `${LOCAL_NAMED_PROFILE} · This device` })).toBeVisible()
    await expect(local).toHaveAttribute('data-reachable', 'true')
    await expect(unreachableMark(page, 'local')).toHaveCount(0)

    // Hold a few seconds: no later roster pass may flip it back.
    await page.waitForTimeout(5_000)
    await expect(local).toHaveAttribute('data-reachable', 'true')
    await expect(unreachableMark(page, 'local')).toHaveCount(0)
  })
})
