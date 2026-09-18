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

  const xvfbRun = (process.env.PATH ?? '').split(delimiter).map(dir => join(dir, 'xvfb-run')).find(existsSync)

  return xvfbRun ? [xvfbRun, '-a'] : null
})()

test.skipIf(displayPrefix === null)(
  'real portal windows handle both providers, transitions and expired/rejected access',
  async () => {
    const root = await mkdtemp(join(tmpdir(), 'hermes-portal-session-'))

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
        if (process.env[name]) {env[name] = process.env[name]}
      }

      const electron: string = createRequire(import.meta.url)('electron')
      const [command, ...args] = [...(displayPrefix ?? []), electron, bundle, root]

      const { stdout } = await promisify(execFile)(command, args, {
        env: { ...env, HERMES_HOME: join(root, '.hermes'), XDG_CONFIG_HOME: join(root, 'config') },
        timeout: 45_000
      })

      expect(stdout).toContain('PORTAL_SESSION_LIVE_OK')
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  },
  60_000
)
