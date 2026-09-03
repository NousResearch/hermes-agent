import * as fs from 'node:fs'
import * as path from 'node:path'

import { buildAppEnv, createSandbox, findElectron } from './fixtures'
import { _electron, type ElectronApplication, expect, test } from './test'

// Exercise module evaluation in the real bundled main process (#101941),
// not the spelling or ordering of declarations in main.ts. Fake boot only
// replaces backend startup; the preference reader and log sinks remain real.
const customLimits = { maxBackends: 5, idleMs: 180_000 }
const desktopRoot = path.resolve(import.meta.dirname, '..')
interface StartupBridge {
  getPoolLimits: () => Promise<{ maxBackends: number; idleMs: number }>
  getRecentLogs: () => Promise<{ path: string; lines: string[] }>
}

const cases = [
  { name: 'missing preference', saved: undefined, env: {}, expected: undefined },
  { name: 'saved preference', saved: JSON.stringify(customLimits), env: {}, expected: customLimits },
  { name: 'invalid JSON', saved: '{invalid', env: {}, expected: undefined },
  {
    name: 'environment fallback',
    saved: undefined,
    env: { HERMES_DESKTOP_POOL_MAX: '5', HERMES_DESKTOP_POOL_IDLE_MS: '180000' },
    expected: customLimits
  }
]

for (const scenario of cases) {
  test(`bundled startup preserves pool limits and early logs: ${scenario.name}`, async () => {
    expect(fs.existsSync(path.join(desktopRoot, 'dist', 'electron-main.mjs')), 'Run npm run build first').toBe(true)
    const sandbox = createSandbox('pool-limits-startup')
    let app: ElectronApplication | undefined

    try {
      if (scenario.saved !== undefined) {
        fs.writeFileSync(path.join(sandbox.userDataDir, 'pool-limits.json'), scenario.saved, 'utf8')
      }

      const env = buildAppEnv(sandbox, {
        HERMES_DESKTOP_BOOT_FAKE: '1',
        HERMES_DESKTOP_BOOT_FAKE_STEP_MS: '120'
      })

      // Isolate fallback settings from the developer/CI shell.
      delete env.HERMES_DESKTOP_POOL_MAX
      delete env.HERMES_DESKTOP_POOL_IDLE_MS
      delete env.HERMES_DESKTOP_DEV_SERVER
      delete env.HERMES_DESKTOP_HERMES
      delete env.HERMES_DESKTOP_HERMES_ROOT
      Object.assign(env, scenario.env)

      app = await _electron.launch({
        executablePath: findElectron(),
        args: [desktopRoot, '--disable-gpu', '--no-sandbox'],
        env,
        timeout: 20_000
      })
      const page = await app.firstWindow()
      await expect(page.locator('#root')).not.toBeEmpty()

      const limits = await page.evaluate(() =>
        (window as typeof window & { hermesDesktop: StartupBridge }).hermesDesktop.getPoolLimits()
      )

      if (scenario.expected) {
        expect(limits).toEqual(scenario.expected)
      } else {
        expect(limits.maxBackends).toBeGreaterThan(0)
        expect(limits.idleMs).toBeGreaterThan(0)
      }

      const recent = await page.evaluate(() =>
        (window as typeof window & { hermesDesktop: StartupBridge }).hermesDesktop.getRecentLogs()
      )

      const earlyLog = recent.lines.find(line => line.includes('[pool-limits]'))
      expect(earlyLog, 'the startup preference log must survive in the in-memory buffer').toBeTruthy()

      // Also exercise the async buffer/timer/promise initialized alongside
      // hermesLog. A guard that silently drops early logs is not a fix.
      const logPath = path.join(sandbox.hermesHome, 'logs', 'desktop.log')
      await expect.poll(() => fs.existsSync(logPath) ? fs.readFileSync(logPath, 'utf8') : '').toContain(earlyLog!)
    } finally {
      await app?.close()
      sandbox.cleanup()
    }
  })
}
