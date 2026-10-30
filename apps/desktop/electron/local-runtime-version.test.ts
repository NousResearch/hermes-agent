import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { PROBE_TIMEOUT_MS } from './backend-probes'
import { parseLocalRuntimeVersion, resolveLocalRuntimeVersion, type VersionProbeRunner } from './local-runtime-version'

describe('local client version source', (): void => {
  it('parses the exact version line emitted by the local launcher', (): void => {
    expect(parseLocalRuntimeVersion('Hermes Agent v0.21.5+6644.gb412ee9 (2026.9.24)\n')).toBe('0.21.5+6644.gb412ee9')
    expect(parseLocalRuntimeVersion('gateway: v0.21.5+6634\n')).toBe('')
  })

  it('probes the installation launcher with a bounded command and ignores remote values', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn(async (_command, args, options) => {
      expect(args).toEqual(['--version'])
      expect(options.cwd).toBe('/isolated/hermes')
      expect(options.timeout).toBe(PROBE_TIMEOUT_MS)
      expect(options.maxBuffer).toBe(64 * 1024)
      expect(options.env.HERMES_HOME).toBe('/isolated/home')

      return 'Hermes Agent v0.21.5+6644.gb412ee9 (local)\n'
    })

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', {
      launcher: '/isolated/hermes/.hermes/bin/hermes',
      run
    })).resolves.toBe('0.21.5+6644.gb412ee9')
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('fails closed when the local launcher cannot answer', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn(async () => {
      throw new Error('timeout')
    })

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', {
      launcher: '/isolated/hermes/.hermes/bin/hermes',
      run
    })).resolves.toBe('')
  })

  it('does not invoke a missing installation launcher', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn()

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', { launcher: null, run })).resolves.toBe('')
    expect(run).not.toHaveBeenCalled()
  })

  // A historical source install (launcher published only to user-bin) proves the launcher's
  // identity with its own `--version`. That probe ran synchronously before the cache check, so
  // every About/focus/profile refresh froze the Electron main thread for the launcher's runtime.
  it.skipIf(process.platform === 'win32')(
    'a historical user-bin launcher never blocks the event loop, first call or cached',
    async (): Promise<void> => {
      const base: string = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hist-launcher-')))

      try {
        const root: string = path.join(base, 'checkout')
        const home: string = path.join(base, 'home')
        fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
        fs.mkdirSync(path.join(root, 'pm'))
        fs.writeFileSync(path.join(root, 'hermes_cli', '_launchers.py'), '')
        fs.mkdirSync(path.join(home, 'bin'), { recursive: true })
        fs.writeFileSync(
          path.join(home, 'bin', 'hermes'),
          `#!/bin/sh\nsleep 1\nprintf '%s\\n' 'Hermes Agent v1.2.3 (2026.10.1)' 'Install directory: ${root}'\n`,
          { mode: 0o755 }
        )

        for (const _call of ['first', 'cached']) {
          let timerFired = false
          const timer = setTimeout(() => { timerFired = true }, 10)
          const startedAt: number = Date.now()
          const pending: Promise<string> = resolveLocalRuntimeVersion(root, home, { isWindows: false })

          expect(Date.now() - startedAt).toBeLessThan(250)
          await new Promise(resolve => setTimeout(resolve, 30))
          expect(timerFired).toBe(true)
          clearTimeout(timer)
          await expect(pending).resolves.toBe('1.2.3')
        }
      } finally {
        fs.rmSync(base, { recursive: true, force: true })
      }
    },
    20_000
  )
})
