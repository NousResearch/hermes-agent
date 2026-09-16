import { type ChildProcess, spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { createSandbox } from './fixtures'
import {
  launchSettingsDesktop,
  seedSettingsProfile,
  type SettingsDesktopWindow,
  settingsEnv
} from './settings-fixtures'
import { expect, test } from './test'

const REPO_ROOT = path.resolve(import.meta.dirname, '../../..')
const REMOTE_ID = 'lab-gateway'
const REMOTE_LABEL = 'Lab gateway'

async function startGateway(env: Record<string, string>): Promise<{ child: ChildProcess; url: string }> {
  const python = ['.venv', 'venv']
    .map(dir => path.join(REPO_ROOT, dir, 'bin', 'python'))
    .find(file => fs.existsSync(file))

  if (!python) {
    throw new Error('A repository Python environment is required for the real-backend test')
  }

  const child = spawn(python, ['-m', 'hermes_cli.main', 'serve', '--host', 'localhost', '--port', '0'], {
    cwd: REPO_ROOT,
    env,
    stdio: ['ignore', 'pipe', 'pipe']
  })

  try {
    const url = await new Promise<string>((resolve, reject) => {
      let output = ''
      const timer = setTimeout(() => reject(new Error('Isolated gateway did not become ready')), 60_000)

      const finish = (error?: Error, port?: string) => {
        clearTimeout(timer)

        if (error) {
          reject(error)
        } else {
          resolve(`http://localhost:${port}`)
        }
      }

      child.on('error', error => finish(error))
      child.on('exit', code => finish(new Error(`Isolated gateway exited: ${code}`)))
      child.stdout?.on('data', chunk => {
        output += String(chunk)
        const ready = output.match(/HERMES_BACKEND_READY port=(\d+)/)

        if (ready) {
          finish(undefined, ready[1])
        }
      })
      child.stderr?.resume()
    })

    return { child, url }
  } catch (error) {
    child.kill('SIGTERM')
    throw error
  }
}

async function stopGateway(child?: ChildProcess) {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>(resolve => {
    const timer = setTimeout(() => child.kill('SIGKILL'), 10_000)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })
    child.kill('SIGTERM')
  })
}

test('Settings selects same-named gateway profiles and edits only the selected owner', async () => {
  test.skip(
    process.platform === 'win32',
    'The real-backend fixture uses a POSIX virtual environment and HOME isolation'
  )
  test.setTimeout(240_000)
  const sandbox = createSandbox('settings-fleet')
  sandbox.hermesHome = path.join(sandbox.root, '.hermes')
  const mock = await startMockServer()
  const remoteRoot = path.join(sandbox.root, 'remote-user')
  const remoteHome = path.join(remoteRoot, '.hermes')
  const sessionToken = randomUUID()
  let remote: Awaited<ReturnType<typeof startGateway>> | undefined
  let app: Awaited<ReturnType<typeof launchSettingsDesktop>>['app'] | undefined

  try {
    seedSettingsProfile(sandbox.hermesHome, mock.url, 25)

    seedSettingsProfile(path.join(sandbox.hermesHome, 'profiles', 'shared'), mock.url, 33)

    seedSettingsProfile(remoteHome, mock.url, 60)
    seedSettingsProfile(path.join(remoteHome, 'profiles', 'shared'), mock.url, 77)
    seedSettingsProfile(path.join(remoteHome, 'profiles', 'remote-only'), mock.url, 78)
    remote = await startGateway({
      ...settingsEnv(sandbox, remoteRoot),
      HERMES_HOME: remoteHome,
      HERMES_DASHBOARD_SESSION_TOKEN: sessionToken
    })
    fs.writeFileSync(
      path.join(sandbox.userDataDir, 'connections.json'),
      JSON.stringify({
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
            url: remote.url,
            authMode: 'token',
            token: { encoding: 'plain', value: sessionToken }
          }
        ]
      })
    )
    const launched = await launchSettingsDesktop(sandbox)
    app = launched.app
    const page = launched.page
    await expect(page.locator('[data-slot="composer-rich-input"]')).toBeVisible({ timeout: 90_000 })

    await page.evaluate(() => {
      window.location.hash = '#/settings?tab=config:model'
    })
    const picker = page.getByRole('button', { name: /^Applies to/ })
    await expect(picker).toBeVisible({ timeout: 30_000 })
    await expect(picker).toContainText('This device')
    await picker.click()
    await expect(page.getByRole('menuitemradio', { name: `shared · ${REMOTE_LABEL}`, exact: true })).toBeVisible({
      timeout: 30_000
    })
    await expect(page.getByRole('menuitemradio', { name: 'shared · This device', exact: true })).toBeVisible()
    await expect(page.getByRole('menuitemradio', { name: `remote-only · ${REMOTE_LABEL}`, exact: true })).toBeVisible()
    await expect(page.getByRole('menuitemradio', { name: 'shared', exact: true })).toHaveCount(0)

    for (const colorScheme of ['dark', 'light'] as const) {
      await page.emulateMedia({ colorScheme })
      await expect
        .poll(() => page.evaluate(() => document.documentElement.classList.contains('dark')))
        .toBe(colorScheme === 'dark')
      await page.screenshot({ path: test.info().outputPath(`settings-fleet-dropdown-${colorScheme}.png`) })
    }

    await page.getByRole('menuitemradio', { name: `shared · ${REMOTE_LABEL}`, exact: true }).click()
    await expect(picker).toContainText(REMOTE_LABEL)
    await expect(page.locator('[data-slot="model-settings-skeleton"]')).toHaveCount(0, { timeout: 30_000 })
    await expect(page.getByRole('button', { name: /^Registered gateways: This device$/ })).toBeVisible()

    const localPath = path.join(sandbox.hermesHome, 'profiles', 'shared', 'config.yaml')
    const localBefore = fs.readFileSync(localPath, 'utf8')
    await page.getByRole('button', { name: 'Advanced', exact: true }).click()
    const maxTurns = page.locator('[data-tour="field-agent.max_turns"] input')
    await expect(maxTurns).toHaveValue('77', { timeout: 30_000 })
    await maxTurns.fill('79')
    await maxTurns.blur()
    await expect
      .poll(async () => {
        const config = await page.evaluate(async () =>
          (window as unknown as SettingsDesktopWindow).hermesDesktop.api<{ agent: { max_turns: number } }>({
            connectionId: 'lab-gateway',
            profile: 'shared',
            path: '/api/config?include_defaults=false'
          })
        )

        // This control can persist a numeric string; ownership is the contract here.
        return String(config.agent.max_turns)
      })
      .toBe('79')
    expect(fs.readFileSync(localPath, 'utf8')).toBe(localBefore)
    await picker.click()
    await page.getByRole('menuitemradio', { name: 'shared · This device', exact: true }).click()
    await expect(maxTurns).toHaveValue('33', { timeout: 30_000 })
    await page.screenshot({ path: test.info().outputPath('settings-local-readback.png') })
  } finally {
    await app?.close()
    await stopGateway(remote?.child)
    await mock.close()
    sandbox.cleanup()
  }
})
