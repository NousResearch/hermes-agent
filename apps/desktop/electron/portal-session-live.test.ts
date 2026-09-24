import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { build } from 'esbuild'
import { expect, test } from 'vitest'

// Chromium needs a display. macOS and Windows always have one; a headless Linux
// runner gets a virtual one through xvfb-run when it is installed (the js-tests
// CI image installs it for this file). Resolved once so the skip is explicit:
// `null` means the test cannot run here, otherwise the command prefix to use.
const displayPrefix = (() => {
  if (process.platform !== 'linux' || process.env.DISPLAY || process.env.WAYLAND_DISPLAY) {
    return []
  }

  const xvfbRun = (process.env.PATH ?? '')
    .split(delimiter)
    .map(dir => join(dir, 'xvfb-run'))
    .find(existsSync)

  return xvfbRun ? [xvfbRun, '-a'] : null
})()

test.skipIf(displayPrefix === null)(
  'real portal windows handle both providers, transitions and expired/rejected access',
  async () => {
    const root = await mkdtemp(join(tmpdir(), 'hermes-portal-session-'))
    let fixtureFailed = false

    try {
      const bundle = join(root, 'main.cjs')
      await build({
        entryPoints: [fileURLToPath(new URL('./portal-session-live-fixture/main.ts', import.meta.url))],
        outfile: bundle,
        bundle: true,
        platform: 'node',
        format: 'cjs',
        external: ['electron']
      })
      const env: NodeJS.ProcessEnv = {}

      for (const name of ['PATH', 'SystemRoot', 'WINDIR', 'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR']) {
        if (process.env[name]) {
          env[name] = process.env[name]
        }
      }

      const electron: string = createRequire(import.meta.url)('electron')
      // Chromium switches go after the fixture's positional root so
      // `process.argv[2]` in the fixture stays the root. `--no-sandbox` matches
      // the Playwright fixture: the npm-installed chrome-sandbox helper is not
      // setuid and Ubuntu 24.04 runners restrict unprivileged user namespaces,
      // so a sandboxed launch aborts before the main script runs.
      const [command, ...args] = [...(displayPrefix ?? []), electron, bundle, root, '--no-sandbox', '--disable-gpu']

      let stdout = ''

      try {
        stdout = (
          await promisify(execFile)(command, args, {
            env: { ...env, HERMES_HOME: join(root, '.hermes'), XDG_CONFIG_HOME: join(root, 'config') },
            // Real Chromium cookie-store transitions took 44s in an isolated
            // Windows run and exceeded 45s under the full 2,500-test suite.
            // Keep a finite process deadline with headroom for native IO.
            timeout: 90_000
          })
        ).stdout
      } catch (error) {
        // execFile's rejection carries only "Command failed"; Electron's real
        // reason (sandbox abort, missing libs, fixture assertion) is on stderr.
        const { stderr = '', stdout: partial = '' } = error as { stderr?: string; stdout?: string }
        throw new Error(`Electron fixture failed.\n--- stdout ---\n${partial}\n--- stderr ---\n${stderr}`, {
          cause: error
        })
      }

      expect(stdout).toContain('PORTAL_SESSION_LIVE_OK')
    } catch (error) {
      fixtureFailed = true
      throw error
    } finally {
      try {
        // Chromium may still hold its Network temp file briefly after Electron
        // is killed by execFile's timeout. Do not hide the fixture's real error.
        await rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
      } catch (cleanupError) {
        if (!fixtureFailed) throw cleanupError
      }
    }
  },
  105_000
)
