/**
 * A generic peer window inherits its initial owner, not a permanent New session
 * target. Exercise two real gateways; only inference uses the shared mock.
 * Requires the built desktop and a repo venv (or HERMES_DESKTOP_PYTHON).
 */
import { type ChildProcess, spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'

import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { collectErrorBanners, type ElectronApplication, expect, installErrorBannerGuard, type Page, test } from './test'

const REPO_ROOT = path.resolve(import.meta.dirname, '../../..')
const LOCAL_LABEL = 'This device'
const REMOTE_LABEL = 'E2E remote'
const REMOTE_ID = 'e2e-remote'
const REMOTE_TOKEN = 'e2e-peer-disposable-token'
const REMOTE_PROFILE = 'ai-dev'

function pythonBinary(): string {
  if (process.env.HERMES_DESKTOP_PYTHON) {
    return process.env.HERMES_DESKTOP_PYTHON
  }

  const suffix = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']
  const candidate = ['.venv', 'venv'].map(dir => path.join(REPO_ROOT, dir, ...suffix)).find(file => fs.existsSync(file))

  if (!candidate) {
    throw new Error('Create the repo Python venv or set HERMES_DESKTOP_PYTHON before running this spec')
  }

  return candidate
}

// Do not inherit profile selection, provider secrets, a dev-server URL, or
// native account state. buildAppEnv supplies the suite's desktop launch flags.
// The temp-dir variables ARE inherited: a sandboxed runner points them at a
// private writable directory, and Electron plus the spawned gateways fall back
// to a possibly unwritable system /tmp without them.
function isolatedEnv(sandbox: Sandbox): Record<string, string> {
  const defaults = buildAppEnv(sandbox)
  const env: Record<string, string> = {}

  for (const key of ['PATH', 'DISPLAY', 'XAUTHORITY', 'LANG', 'LC_ALL', 'SystemRoot', 'COMSPEC', 'PATHEXT', 'TMPDIR', 'TEMP', 'TMP']) {
    if (defaults[key]) {
      env[key] = defaults[key]
    }
  }

  for (const key of ['XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR', 'APPDATA', 'LOCALAPPDATA']) {
    env[key] = path.join(sandbox.root, key.toLowerCase())
    fs.mkdirSync(env[key], { recursive: true, mode: 0o700 })
  }

  for (const key of ['HERMES_DESKTOP_USER_DATA_DIR', 'HERMES_DESKTOP_IGNORE_EXISTING', 'HERMES_DESKTOP_HERMES_ROOT', 'HERMES_DESKTOP_APP_NAME', 'HERMES_DESKTOP_SKIP_QUIT_CONFIRM']) {
    env[key] = defaults[key]
  }

  return {
    ...env,
    HOME: sandbox.root,
    USERPROFILE: sandbox.root,
    HERMES_HOME: sandbox.hermesHome,
    HERMES_DESKTOP_PYTHON: pythonBinary(),
    PYTHONPATH: REPO_ROOT,
    // On Linux CI, DISPLAY belongs to Xvfb rather than the host Wayland seat.
    ...(process.platform === 'linux' ? { XDG_SESSION_TYPE: 'x11', ELECTRON_OZONE_PLATFORM_HINT: 'x11' } : {}),
  }
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(error => error ? reject(error) : resolve(port))
    })
  })
}

function remoteAlive(child: ChildProcess): boolean {
  return Boolean(child.pid) && child.exitCode === null && child.signalCode === null
}

// The remote runs in its own process group so its Python children die with it.
// ESRCH (group already gone) must not abort the rest of teardown.
function signalRemote(child: ChildProcess, signal: NodeJS.Signals): void {
  try {
    if (process.platform === 'win32') {
      child.kill(signal)
    } else {
      process.kill(-child.pid!, signal)
    }
  } catch {
    /* already exited */
  }
}

async function stopRemote(child: ChildProcess | undefined): Promise<void> {
  if (!child || !remoteAlive(child)) {
    return
  }

  const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))
  signalRemote(child, 'SIGTERM')
  let timer: ReturnType<typeof setTimeout> | undefined
  await Promise.race([exited, new Promise<void>(resolve => { timer = setTimeout(resolve, 5_000) })])
  clearTimeout(timer)

  if (remoteAlive(child)) {
    signalRemote(child, 'SIGKILL')
    await exited
  }
}

const gateway = (page: Page) => page.getByRole('button', { name: /^Registered gateways: / })
const composer = (page: Page) => page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()

async function expectReady(page: Page, label: string): Promise<void> {
  await expect(gateway(page)).toHaveAttribute('aria-label', `Registered gateways: ${label}`, { timeout: 90_000 })
  await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 90_000 })
  await expect(composer(page)).toBeEditable()
}

async function switchTo(page: Page, label: string): Promise<void> {
  await gateway(page).click()
  await page.getByRole('menuitemradio', { name: new RegExp(label) }).click()
  await expectReady(page, label)
}

async function newSession(page: Page, label: string): Promise<void> {
  await page.locator('[data-tour="sidebar-nav-new-session"]').click()
  await expectReady(page, label)
  // Query the menu after the reset as well: the checked connection must agree
  // with the ready statusbar, not merely retain its pre-click text for a frame.
  await gateway(page).click()
  await expect(page.getByRole('menuitemradio', { name: new RegExp(label) })).toHaveAttribute('aria-checked', 'true')
  await page.keyboard.press('Escape')
}

const peerTest = test.extend<{ gateways: { app: ElectronApplication; source: Page } }>({
  // Playwright requires object destructuring even for a fixture with no dependencies.
  // eslint-disable-next-line no-empty-pattern
  gateways: async ({}, runTest, info) => {
    const local = createSandbox('peer-local')
    const remote = createSandbox('peer-remote')
    const mock = await startMockServer()
    let child: ChildProcess | undefined
    let app: ElectronApplication | undefined
    let remoteLog = ''
    let desktopLog = ''

    try {
      const remoteProfileHome = path.join(remote.hermesHome, 'profiles', REMOTE_PROFILE)
      fs.mkdirSync(remoteProfileHome, { recursive: true })

      for (const hermesHome of [local.hermesHome, remote.hermesHome, remoteProfileHome]) {
        writeMockProviderConfig(hermesHome, mock.url)
        writeEnvFile(hermesHome)
      }

      const port = await freePort()
      const remoteUrl = `http://127.0.0.1:${port}`
      child = spawn(pythonBinary(), ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'], {
        cwd: REPO_ROOT,
        detached: process.platform !== 'win32',
        env: { ...isolatedEnv(remote), HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN },
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      // Fixture setup shares the test budget; a timeout while still awaiting
      // readiness abandons this function before `finally`, and a detached
      // child outlives the worker. Reap it on process exit regardless.
      const spawned = child
      process.once('exit', () => {
        if (remoteAlive(spawned)) {
          signalRemote(spawned, 'SIGKILL')
        }
      })
      child.stdout?.on('data', chunk => { remoteLog += chunk.toString() })
      child.stderr?.on('data', chunk => { remoteLog += chunk.toString() })
      let spawnError: Error | undefined
      child.on('error', error => { spawnError = error })
      await expect.poll(async () => {
        if (spawnError) { throw spawnError }

        if (child!.exitCode !== null) { throw new Error(`Remote exited: ${remoteLog}`) }

        try {
          return (await fetch(`${remoteUrl}/api/status`, {
            headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN },
            signal: AbortSignal.timeout(2_000),
          })).status
        } catch { return 0 }
      }, { timeout: 90_000 }).toBe(200)

      fs.writeFileSync(path.join(local.userDataDir, 'connections.json'), JSON.stringify({
        version: 2, primary: 'local', launchMode: 'primary', lastUsed: 'local',
        connections: [
          { id: 'local', kind: 'local', label: LOCAL_LABEL },
          { id: REMOTE_ID, kind: 'remote', label: REMOTE_LABEL, url: remoteUrl,
            authMode: 'token', token: { encoding: 'plain', value: REMOTE_TOKEN } },
        ],
      }), { encoding: 'utf8', mode: 0o600 })

      const launched = await launchDesktop(isolatedEnv(local))
      app = launched.app
      app.process().stdout?.on('data', chunk => { desktopLog += chunk.toString() })
      app.process().stderr?.on('data', chunk => { desktopLog += chunk.toString() })
      await expectReady(launched.page, LOCAL_LABEL)
      await runTest({ app, source: launched.page })
    } finally {
      const errors = new Set<string>()

      if (app) {
        for (const page of app.windows()) {
          for (const error of await collectErrorBanners(page)) {
            errors.add(error)
          }

          if (info.status !== info.expectedStatus) {
            await page.screenshot({ path: info.outputPath(`window-${app.windows().indexOf(page)}.png`) }).catch(() => undefined)
          }
        }

        await app.close().catch(() => undefined)
      }

      await stopRemote(child)
      fs.writeFileSync(info.outputPath('remote.log'), remoteLog)
      fs.writeFileSync(info.outputPath('desktop.log'), desktopLog)
      await mock.close()
      local.cleanup()
      remote.cleanup()
      // The shared afterEach checks only the last guarded page and runs before
      // fixture teardown. Assert every window here, after releasing resources.
      expect([...errors], 'Error banners in source or peer window').toEqual([])
    }
  },
})

peerTest.setTimeout(240_000)

peerTest('Ctrl+Shift+N resumes a remote-only profile on its owning gateway', async ({ gateways: { app, source } }) => {
  const remoteProfile = source.locator(`[data-slot="profile-rail-gateway"][data-connection-id="${REMOTE_ID}"]`)
    .getByRole('button', { name: `${REMOTE_PROFILE} · ${REMOTE_LABEL}`, exact: true })

  await expect(remoteProfile).toBeVisible({ timeout: 60_000 })
  await remoteProfile.click()
  await expectReady(source, REMOTE_LABEL)
  await composer(source).fill('Remember this remote profile session for the peer-window regression.')
  await composer(source).press('Enter')
  await expect(source.locator('[data-slot="aui_assistant-message-content"]').getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 60_000 })

  const opened = app.waitForEvent('window')
  await source.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+N' : 'Control+Shift+N')
  const peer = await opened
  installErrorBannerGuard(peer)
  await expectReady(peer, REMOTE_LABEL)
  await expect(peer.locator('[data-slot="aui_assistant-message-content"]').getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 60_000 })
  await expectReady(source, REMOTE_LABEL)
  await peer.reload()
  await expectReady(peer, REMOTE_LABEL)
  await expect(peer.locator('[data-slot="aui_assistant-message-content"]').getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 60_000 })
  await composer(peer).fill('Continue this session after reopening the peer window.')
  await composer(peer).press('Enter')
  await expect(peer.locator('[data-slot="aui_assistant-message-content"]').getByText(MOCK_REPLY, { exact: true })).toHaveCount(2, { timeout: 60_000 })
  await expectReady(source, REMOTE_LABEL)
})

peerTest('Ctrl+Shift+N peer keeps local → remote → local choices on New session', async ({ gateways: { app, source } }) => {
  const sourceDraft = 'Keep this source-window draft'
  await composer(source).fill(sourceDraft)
  const opened = app.waitForEvent('window')
  await source.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+N' : 'Control+Shift+N')
  const peer = await opened
  installErrorBannerGuard(peer)
  await expectReady(peer, LOCAL_LABEL)

  for (const label of [LOCAL_LABEL, REMOTE_LABEL, LOCAL_LABEL]) {
    await test.step(`New session stays on ${label}`, async () => {
      await switchTo(peer, label)
      await newSession(peer, label)
      await expectReady(source, LOCAL_LABEL)
      await expect(composer(source)).toHaveText(sourceDraft)
    })
  }
})

peerTest('explicit profile window returns to its boot owner on New session', async ({ gateways: { app, source } }) => {
  const remoteProfile = source.locator(`[data-slot="profile-rail-gateway"][data-connection-id="${REMOTE_ID}"]`)
    .getByRole('button', { name: `default · ${REMOTE_LABEL}`, exact: true })

  await expect(remoteProfile).toBeVisible({ timeout: 60_000 })
  await remoteProfile.click({ button: 'right' })
  const opened = app.waitForEvent('window')
  await source.getByRole('menuitem', { name: 'Open in new window', exact: true }).click()
  const pinned = await opened
  installErrorBannerGuard(pinned)
  await expectReady(pinned, REMOTE_LABEL)
  await switchTo(pinned, LOCAL_LABEL)
  await newSession(pinned, REMOTE_LABEL)
  await expectReady(source, LOCAL_LABEL)
})
