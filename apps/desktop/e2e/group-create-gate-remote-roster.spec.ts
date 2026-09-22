/**
 * E2E: the New Group Chat menu gate with one local bot plus remote-connection
 * bots (#101543). "This device" is the Electron-managed local backend with
 * only its `default` profile; "Homelab" is a REAL second `hermes serve`
 * registered as a remote URL connection. The gate must count the same
 * selectable set the dialog seats — bots from every registered connection —
 * so 1 local + 1 remote enables New Group Chat and the room can be created.
 */

import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

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
import { type ElectronApplication, expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const REMOTE_LABEL = 'Homelab'
const REMOTE_ID = 'homelab'
const REMOTE_TOKEN = 'e2e-group-gate-homelab-token'
const SHOTS = path.join(os.tmpdir(), 'batchbots/group-identity-members/shots')

interface RemoteGateway {
  url: string
  close: () => Promise<void>
}

/** The worktree's own backend: `python -m hermes_cli.main` from the repo root
 *  (the venv's `hermes` console script resolves the package it was installed
 *  from, which need not be this checkout). */
function hermesCommand(): { bin: string; args: string[] } {
  const venvPython = path.join(REPO_ROOT, '.venv', 'bin', 'python')

  if (fs.existsSync(venvPython)) {
    return { bin: venvPython, args: ['-m', 'hermes_cli.main'] }
  }

  const result = spawnSync('which', ['hermes'], { encoding: 'utf8' })

  if (result.status === 0 && result.stdout.trim()) {
    return { bin: result.stdout.trim(), args: [] }
  }

  throw new Error('hermes backend not found: create the repo venv (uv sync) or put hermes on PATH')
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

  const hermes = hermesCommand()

  const child: ChildProcess = spawn(hermes.bin, [...hermes.args, 'serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'], {
    cwd: REPO_ROOT,
    detached: true,
    env: { ...process.env, HERMES_HOME: home, HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN, PYTHONPATH: REPO_ROOT },
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  let log = ''
  child.stdout?.on('data', (chunk: Buffer) => { log += chunk.toString() })
  child.stderr?.on('data', (chunk: Buffer) => { log += chunk.toString() })
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
          { id: REMOTE_ID, kind: 'remote', label: REMOTE_LABEL, url: remoteUrl, authMode: 'token', token: { encoding: 'plain', value: REMOTE_TOKEN } },
        ],
      },
      null,
      2,
    ),
    { encoding: 'utf8', mode: 0o600 },
  )
}

const roster = (page: Page) => page.locator('[data-slot="bots-roster"]')

test.describe('New Group Chat gate — one local bot plus remote-connection bots', () => {
  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let remote: RemoteGateway
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    mock = await startMockServer()
    sandbox = createSandbox('group-gate')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    // This device: ONLY its primary `default` profile. Homelab: default + inbox (rendered title-cased, "Inbox").
    remote = await startRemoteGateway(sandbox.root, mock.url, ['inbox'])
    writeConnectionsRegistry(sandbox, remote.url)
    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await remote?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('the menu entry enables and the dialog seats the remote bot', async () => {
    test.setTimeout(240_000)
    const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
    await tab.click()
    await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
    // Precondition: exactly one local bot, and the remote bots have arrived.
    await expect(roster(page).locator(`[data-roster-key="${REMOTE_ID}::inbox"]`)).toBeVisible({ timeout: 90_000 })
    await expect(roster(page).locator('[data-roster-key^="local::"]')).toHaveCount(1)

    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    const entry = page.getByRole('menuitem', { name: 'New Group Chat' })
    await expect(entry).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/group-create-gate-menu.png` })
    await expect(entry).toBeEnabled()
    await entry.click()

    const dialog = page.getByRole('dialog', { name: 'New Group Chat' })
    await expect(dialog).toBeVisible()
    await dialog.getByText('Hermes', { exact: true }).first().locator('xpath=ancestor::label').getByRole('checkbox').click()
    await dialog.getByText('Inbox', { exact: true }).first().locator('xpath=ancestor::label').getByRole('checkbox').click()
    await expect(dialog.getByRole('button', { name: 'Create Group (2)' })).toBeEnabled()
    await page.screenshot({ path: `${SHOTS}/group-create-gate-dialog.png` })
  })
})
