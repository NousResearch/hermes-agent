/** Native Electron main -> real headless serve, without an app/renderer build.
 * Opt in with HERMES_TEST_REAL_SERVE=1; see pool-retirement-live-fixture/README.md.
 */
import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { test } from 'vitest'

const require = createRequire(import.meta.url)

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repo = resolve(desktop, '../..')
const fixture = join(desktop, 'electron/pool-retirement-live-fixture')

function isolatedEnv(root: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}

  // Allowlist rather than trying to enumerate provider secrets and overrides.
  for (const name of ['PATH', 'SystemRoot', 'WINDIR', 'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR']) {
    if (process.env[name]) {
      env[name] = process.env[name]
    }
  }

  return {
    ...env,
    HOME: root,
    USERPROFILE: root,
    HERMES_HOME: join(root, '.hermes'),
    XDG_CONFIG_HOME: join(root, 'config'),
    XDG_CACHE_HOME: join(root, 'cache'),
    TMPDIR: root,
    TEMP: root,
    TMP: root,
    TZ: 'UTC',
    LANG: 'C.UTF-8',
    HERMES_DESKTOP_CDP_PORT: 'off',
    HERMES_DESKTOP_USER_DATA_DIR: join(root, 'user-data')
  }
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<number | null> {
  return new Promise((resolveExit, reject) => {
    const timer = setTimeout(() => reject(new Error('Native retirement fixture timed out')), timeoutMs)
    child.once('error', error => {
      clearTimeout(timer)
      reject(error)
    })
    child.once('exit', code => {
      clearTimeout(timer)
      resolveExit(code)
    })
  })
}

test.skipIf(process.env.HERMES_TEST_REAL_SERVE !== '1' || process.platform === 'win32')(
  'native retirement preserves backend-only cron and hands both waiters capacity only after real child exit',
  async () => {
    const python = process.env.HERMES_TEST_PYTHON
    assert.ok(python && existsSync(python), 'Set HERMES_TEST_PYTHON to an installed Hermes Python environment')
    const electron = process.env.HERMES_TEST_ELECTRON || (require('electron') as string)
    assert.ok(existsSync(electron), 'HERMES_TEST_ELECTRON must name a real native Electron executable')
    const root = mkdtempSync(join(tmpdir(), 'hermes-pool-retirement-live-'))
    const resultPath = join(root, 'result.json')
    let child: ChildProcess | undefined
    let output = ''

    try {
      mkdirSync(join(root, '.hermes'))
      const bundle = join(root, 'main.cjs')
      await build({
        entryPoints: [join(fixture, 'main.ts')],
        outfile: bundle,
        bundle: true,
        platform: 'node',
        format: 'cjs',
        target: 'node22',
        external: ['electron']
      })
      await build({
        entryPoints: [join(fixture, 'preload.ts')],
        outfile: join(root, 'preload.cjs'),
        bundle: true,
        platform: 'node',
        format: 'cjs',
        external: ['electron']
      })
      await build({
        entryPoints: [join(desktop, 'src/test/pool-retirement-renderer.ts')],
        outfile: join(root, 'renderer.js'),
        bundle: true,
        platform: 'browser',
        format: 'iife',
        alias: { '@': join(desktop, 'src'), '@hermes/shared': join(repo, 'apps/shared/src') },
        define: { 'import.meta.env': '{}', 'import.meta.hot': 'undefined' }
      })
      writeFileSync(
        join(root, 'renderer.html'),
        '<!doctype html><meta charset="utf-8"><title>Retirement seam fixture</title><script src="./renderer.js"></script>'
      )
      child = spawn(electron, [bundle, root, repo, python, fixture], {
        cwd: root,
        env: isolatedEnv(root),
        stdio: ['ignore', 'pipe', 'pipe']
      })
      // Capture the OS handle now; do not ask a disposed Electron dispatcher after quit.
      child.stdout!.on('data', chunk => {
        output += String(chunk)
      })
      child.stderr!.on('data', chunk => {
        output += String(chunk)
      })
      const code = await waitForExit(child, 180_000)
      assert.ok(existsSync(resultPath), `No native result (exit ${code}):\n${output}`)
      const result = JSON.parse(readFileSync(resultPath, 'utf8'))
      process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
      assert.equal(code, 0, `${result.error || 'Native fixture failed'}\n${output}`)
      assert.equal(result.ok, true)
      assert.ok(result.electronVersion, 'Must run native Electron, not ELECTRON_RUN_AS_NODE')
      assert.equal(result.processType, 'browser')
      assert.equal(result.maxLiveServeChildren, result.capacity)
      assert.deepEqual(result.retired, ['idle-a', 'idle-b'])
      assert.deepEqual(result.opened, ['foreground-d', 'promoted-e'])
      assert.equal(result.busyCronSurvived, true)
      assert.equal(result.rendererParking.redials, 0)
      assert.deepEqual(result.rendererParking.parked['idle-a'], ['idle-a', 'conn:local::idle-a'])
      assert.deepEqual(result.rendererParking.parked['idle-b'], ['idle-b', 'conn:local::idle-b'])
      assert.equal(result.cleanedUp, true)
    } finally {
      if (child && child.exitCode === null && child.signalCode === null) {
        child.kill('SIGTERM')
        await waitForExit(child, 15_000).catch(async () => {
          child!.kill('SIGKILL')
          await waitForExit(child!, 5000)
        })
      }

      rmSync(root, { recursive: true, force: true })
    }
  },
  210_000
)
