/** Real Electron argv/single-instance/preload/renderer path. Only the final
 * update effects are replaced: this spec must never update its own checkout. */
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, findElectron } from './fixtures'
import { _electron, type ElectronApplication, expect, installErrorBannerGuard, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')

function requestFile(pid: number, id: string) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-all-'))
  const file = path.join(directory, 'request.json')
  const value = { version: 1, id, state: 'pending', launcher_pids: [pid], expires_at: Date.now() + 59_000 }
  fs.writeFileSync(file, JSON.stringify(value))

  return { directory, file, value }
}

test('cold launch dispatches once; a running-instance request coalesces and reload does not replay', async () => {
  test.setTimeout(120_000)
  const sandbox = createSandbox('update-all-launch')
  const mock = await startMockServer()
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  const env = buildAppEnv(sandbox)
  // Real process liveness gates the action until the safe updater doubles are installed.
  const launcher = spawn(process.execPath, ['-e', 'process.stdin.resume()'], { stdio: ['pipe', 'ignore', 'ignore'] })
  const first = requestFile(launcher.pid!, 'a'.repeat(32))
  const duplicate = requestFile(process.pid, 'b'.repeat(32))
  let app: ElectronApplication | undefined

  try {
    app = await _electron.launch({
      executablePath: findElectron(),
      args: [DESKTOP_ROOT, '--disable-gpu', '--no-sandbox', `--hermes-update-all-request=${first.file}`],
      env,
      cwd: DESKTOP_ROOT
    })
    const page = await app.firstWindow()
    installErrorBannerGuard(page)
    await expect.poll(() => JSON.parse(fs.readFileSync(first.file, 'utf8')).state).toBe('accepted')
    expect(await app.evaluate(() => process.argv.some(arg => arg.startsWith('--hermes-update-all-request=')))).toBe(
      false
    )
    await app.evaluate(({ ipcMain }) => {
      const state = globalThis as unknown as { updateOrder: string[]; finishRemote: () => void }
      state.updateOrder = []
      ipcMain.removeHandler('hermes:connections:list')
      ipcMain.handle('hermes:connections:list', () => ({
        version: 2,
        primary: 'local',
        lastUsed: 'local',
        launchMode: 'primary',
        connections: [
          { id: 'local', kind: 'local', label: 'Local' },
          { id: 'test-remote', kind: 'remote', label: 'Remote' }
        ]
      }))
      ipcMain.removeHandler('hermes:connections:update-all')
      ipcMain.handle('hermes:connections:update-all', () => {
        state.updateOrder.push('remote')

        return new Promise(resolve => {
          state.finishRemote = () => resolve({ ok: true, results: [] })
        })
      })
      ipcMain.removeHandler('hermes:updates:check')
      ipcMain.handle('hermes:updates:check', () => ({ supported: true, behind: 1, updateAvailable: true }))
      ipcMain.removeHandler('hermes:updates:apply')
      ipcMain.handle('hermes:updates:apply', () => {
        state.updateOrder.push('local')

        return { ok: true }
      })
    })
    const staged = first.file.replace(/\.json$/, '.tmp')
    fs.writeFileSync(staged, JSON.stringify({ ...first.value, state: 'committed' }))
    fs.renameSync(staged, first.file)
    const exited = once(launcher, 'exit')
    launcher.stdin!.end()
    await exited
    const order = () => app!.evaluate(() => (globalThis as unknown as { updateOrder: string[] }).updateOrder)
    await expect.poll(order, { timeout: 30_000 }).toEqual(['remote'])

    const secondary = spawn(
      findElectron(),
      [DESKTOP_ROOT, '--no-sandbox', `--hermes-update-all-request=${duplicate.file}`],
      {
        cwd: DESKTOP_ROOT,
        env,
        stdio: 'ignore'
      }
    )

    const [code] = await once(secondary, 'exit')
    expect(code).toBe(0)
    await expect.poll(() => JSON.parse(fs.readFileSync(duplicate.file, 'utf8')).state).toBe('coalesced')
    expect(await order()).toEqual(['remote'])
    await app.evaluate(() => (globalThis as unknown as { finishRemote: () => void }).finishRemote())
    await expect.poll(order, { timeout: 45_000 }).toEqual(['remote', 'local'])
    await page.reload()
    await page.waitForFunction(
      () =>
        typeof (window as Window & { hermesDesktop?: { onUpdateAllRequested?: unknown } }).hermesDesktop
          ?.onUpdateAllRequested === 'function'
    )
    expect(await order()).toEqual(['remote', 'local'])
  } finally {
    launcher.stdin?.end()
    await app?.close().catch(() => undefined)
    await mock.close()
    sandbox.cleanup()
    fs.rmSync(first.directory, { recursive: true, force: true })
    fs.rmSync(duplicate.directory, { recursive: true, force: true })
  }
})
