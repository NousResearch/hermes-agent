import { spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { expect, test } from 'vitest'

const require = createRequire(import.meta.url)

test('native Chromium: no-reset control stalls, wake cleanup recovers on a fresh socket without losing login', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'hermes-wake-native-'))

  try {
    const bundle = path.join(root, 'fixture.cjs')
    await build({
      entryPoints: [fileURLToPath(new URL('./wake-transport-native-fixture.ts', import.meta.url))],
      outfile: bundle,
      bundle: true,
      platform: 'node',
      format: 'cjs',
      target: 'node22',
      external: ['electron']
    })
    const electron = require('electron') as string
    const env: NodeJS.ProcessEnv = {}

    for (const name of ['PATH', 'SystemRoot', 'WINDIR', 'TMPDIR', 'TEMP', 'TMP', 'XDG_RUNTIME_DIR']) {
      if (process.env[name]) {
        env[name] = process.env[name]
      }
    }

    const run = (mode: string) =>
      new Promise<{ code: number | null; output: string }>((resolve, reject) => {
        // The JS CI lane installs xvfb; run a private display rather than depend
        // on a developer's desktop or skip Chromium coverage in headless CI.
        const command = process.platform === 'linux' ? 'xvfb-run' : electron

        // Linux CI can prohibit unprivileged user namespaces. Only this
        // no-window, loopback-only fixture opts out of Chromium's sandbox.
        const args = [
          ...(process.platform === 'linux' ? ['-a', electron] : []),
          bundle,
          root,
          mode,
          ...(process.platform === 'linux' ? ['--no-sandbox'] : [])
        ]

        const child = spawn(command, args, {
          cwd: root,
          env,
          detached: process.platform !== 'win32',
          stdio: ['ignore', 'pipe', 'pipe']
        })

        let output = ''
        let timedOut = false

        const timer = setTimeout(() => {
          timedOut = true

          // Kill the private POSIX process group, not just xvfb-run's shell.
          // Otherwise a timed-out fixture can orphan Xvfb and Electron.
          if (process.platform !== 'win32' && child.pid) {
            try {
              process.kill(-child.pid, 'SIGKILL')
            } catch (error) {
              if ((error as NodeJS.ErrnoException).code !== 'ESRCH') {
                reject(error)
              }
            }
          } else {
            child.kill('SIGKILL')
          }
        }, 30_000)

        child.stdout.on('data', chunk => {
          output += chunk
        })
        child.stderr.on('data', chunk => {
          output += chunk
        })
        child.on('error', error => {
          clearTimeout(timer)
          reject(error)
        })
        child.on('close', code => {
          clearTimeout(timer)

          if (timedOut) {
            reject(new Error(`Native wake fixture timed out:\n${output}`))
          } else {
            resolve({ code, output })
          }
        })
      })

    const baseline = await run('baseline')
    expect(baseline.code, baseline.output).not.toBe(0)
    expect(baseline.output).toContain('STALE_POOLED_SOCKET_REUSED')
    const patched = await run('patched')
    expect(patched.code, patched.output).toBe(0)
    const marker = patched.output.split('\n').find(line => line.startsWith('WAKE_NATIVE_RESULT '))
    expect(marker, patched.output).toBeDefined()
    const result = JSON.parse(marker!.slice('WAKE_NATIVE_RESULT '.length))
    expect(result).toMatchObject({ ok: true, processType: 'browser', freshSocket: true, cookiePreserved: true })
    expect(result.electron).toBeTruthy()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}, 90_000)
