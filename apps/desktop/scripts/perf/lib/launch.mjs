// Connect the harness to a renderer — either an already-running debug instance
// (`attach`) or a freshly spawned, fully isolated one (`startIsolatedInstance`).
//
// The isolated instance is what makes the harness self-contained and unblocks
// the measurement that the single-instance lock used to prevent:
//   · its own --user-data-dir  → its own Electron single-instance lock, so it
//     never collides with (or steals focus from) the user's running `hgui`.
//   · its own HERMES_HOME      → its own backend + sessions, no shared state.
//   · its own --remote-debugging-port → a private CDP endpoint.
//   · HERMES_DESKTOP_BOOT_FAKE=1 → deterministic boot overlay.
// The synthetic scenarios drive `$messages` directly, so no LLM credits are
// spent regardless of the isolated backend.

import { spawn } from 'node:child_process'
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import { homedir, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { CDP, requireDriver, sleep } from './cdp.mjs'

const require = createRequire(import.meta.url)
const DESKTOP_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

async function reachable(url) {
  try {
    await fetch(url)

    return true
  } catch {
    return false
  }
}

async function waitFor(fn, { timeoutMs, label }) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await fn()) {
      return
    }

    await sleep(300)
  }

  throw new Error(`timed out after ${timeoutMs}ms waiting for ${label}`)
}

// Seed an isolated HERMES_HOME with just enough config (NOT sessions) so the
// spawned instance reaches an empty chat view instead of the onboarding wizard.
// A separate HERMES_HOME dir means a separate gateway lock — no collision with
// the user's running app, which keeps its own sessions DB and state.
function seedConfigFrom(sourceHome, targetHome) {
  if (!existsSync(sourceHome)) {
    return
  }

  for (const name of ['config.yaml', '.env', 'auth.json']) {
    const from = join(sourceHome, name)

    if (existsSync(from)) {
      try {
        copyFileSync(from, join(targetHome, name))
      } catch {
        // best-effort — a missing file just means onboarding may appear.
      }
    }
  }
}

// Resolve the vite CLI entry via its package.json `bin` (Vite 8's `exports`
// blocks importing `vite/bin/vite.js` directly).
function resolveViteBin() {
  const pkgPath = require.resolve('vite/package.json')
  const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'))
  const rel = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin?.vite

  if (!rel) {
    throw new Error('could not resolve the vite CLI from vite/package.json')
  }

  return join(dirname(pkgPath), rel)
}

// Poll the perf driver's `connected()` until the gateway socket is open.
// Returns false if the probe predates this helper or the timeout elapses.
async function waitForConnected(cdp, timeoutMs) {
  const hasProbe = await cdp.eval('typeof window.__PERF_DRIVE__.connected === "function"')

  if (!hasProbe) {
    return false
  }

  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await cdp.eval('window.__PERF_DRIVE__.connected()')) {
      return true
    }

    await sleep(500)
  }

  return false
}

function runProcess(command, args, { env } = {}) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, {
      cwd: DESKTOP_DIR,
      stdio: 'inherit',
      env: env ? { ...process.env, ...env } : process.env
    })
    child.on('error', reject)
    child.on('exit', code => (code === 0 ? resolveRun() : reject(new Error(`${command} ${args[0]} exited ${code}`))))
  })
}

function runNode(scriptRelPath, args = []) {
  return runProcess(process.execPath, [join(DESKTOP_DIR, scriptRelPath), ...args])
}

// Build a production renderer WITH the perf probe included (VITE_PERF_PROBE=1),
// plus the prod electron-main bundle, so the harness can measure a real,
// minified React build instead of the ~3x-slower dev build. Slow (a full vite
// build); do it once, then run/attach many times.
export async function buildProdRenderer() {
  const viteBin = resolveViteBin()
  await runProcess(process.execPath, [viteBin, 'build'], { env: { VITE_PERF_PROBE: '1' } })
  await runNode('scripts/bundle-electron-main.mjs')
}

/** Attach to a renderer already listening on `port` (launched via perf:serve or with --remote-debugging-port). */
export async function attach({ port = 9222, match } = {}) {
  const cdp = await CDP.connect({ port, match })
  await requireDriver(cdp)

  return { cdp, teardown: () => cdp.close() }
}

/**
 * Spawn an isolated dev instance (vite + electron), wait for the perf driver,
 * and return `{ cdp, teardown, devUrl, port }`. `teardown` kills both children
 * and removes any temp dirs it created.
 */
// Chromium switches that stop frame-production throttling for a window that
// isn't foregrounded (the perf window usually sits behind the IDE/terminal).
const ANTI_THROTTLE_FLAGS = [
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--disable-features=CalculateNativeWinOcclusion'
]

/**
 * Spawn an isolated instance and connect the perf driver. Two render modes:
 *   · dev (default): vite dev server + dev electron-main bundle.
 *   · prod (`prod: true`): a production build (call buildProdRenderer first);
 *     electron loads dist/index.html — representative, minified React.
 * `coldStart: true` skips the gateway-connect wait and settle (for launch-time
 * measurement) and returns `timings` (spawn→CDP, spawn→driver) plus renderer
 * boot marks (FCP, time-to-composer).
 */
export async function startIsolatedInstance({
  port = 9222,
  devPort = 5174,
  prod = false,
  coldStart = false,
  hermesHome,
  userDataDir,
  seedConfig = true,
  bootFakeStepMs = 120,
  settleMs = 2500,
  connectTimeoutMs = 90000
} = {}) {
  const children = []
  const tempDirs = []

  const mkTemp = prefix => {
    const dir = mkdtempSync(join(tmpdir(), prefix))
    tempDirs.push(dir)

    return dir
  }

  const home = hermesHome ?? mkTemp('hermes-perf-home-')
  const userData = userDataDir ?? mkTemp('hermes-perf-ud-')
  const devUrl = prod ? null : `http://127.0.0.1:${devPort}`

  if (seedConfig && !hermesHome) {
    seedConfigFrom(join(homedir(), '.hermes'), home)
  }

  const teardown = () => {
    for (const child of children) {
      try {
        child.kill('SIGTERM')
      } catch {
        // already gone
      }
    }

    for (const dir of tempDirs) {
      try {
        rmSync(dir, { recursive: true, force: true })
      } catch {
        // best-effort
      }
    }
  }

  try {
    if (prod) {
      // Renderer + main are expected pre-built (buildProdRenderer). Cheap to
      // re-bundle main so an isolated run always matches current source.
      await runNode('scripts/bundle-electron-main.mjs')
    } else {
      if (!(await reachable(devUrl))) {
        const viteBin = resolveViteBin()
        const vite = spawn(process.execPath, [viteBin, '--host', '127.0.0.1', '--port', String(devPort)], {
          cwd: DESKTOP_DIR,
          stdio: ['ignore', 'inherit', 'inherit']
        })
        children.push(vite)
        await waitFor(() => reachable(devUrl), { timeoutMs: 60000, label: `vite dev server on :${devPort}` })
      }

      await runNode('scripts/bundle-electron-main.mjs', ['--dev'])
    }

    // Isolated Electron: own --user-data-dir (single-instance lock scope) + own
    // HERMES_HOME (backend + sessions). No DEV_SERVER env in prod → dist load.
    const electronBin = require('electron')
    const env = {
      ...process.env,
      HERMES_HOME: home,
      HERMES_DESKTOP_BOOT_FAKE: '1',
      HERMES_DESKTOP_BOOT_FAKE_STEP_MS: String(bootFakeStepMs),
      XCURSOR_SIZE: '24'
    }

    if (devUrl) {
      env.HERMES_DESKTOP_DEV_SERVER = devUrl
    }

    const spawnAt = Date.now()
    const electron = spawn(
      electronBin,
      ['.', `--user-data-dir=${userData}`, `--remote-debugging-port=${port}`, ...ANTI_THROTTLE_FLAGS],
      { cwd: DESKTOP_DIR, stdio: ['ignore', 'inherit', 'inherit'], env }
    )
    children.push(electron)

    // Wait for the renderer + perf driver. In prod the target URL is file://,
    // so don't match on the dev port.
    let cdp = null
    let cdpAt = 0
    await waitFor(
      async () => {
        try {
          cdp = await CDP.connect({ port, match: devUrl ? String(devPort) : undefined, timeoutMs: 2000 })
          cdpAt = cdpAt || Date.now()

          return await cdp.eval('!!(window.__PERF_DRIVE__ && window.__PERF_DRIVE__.stream)')
        } catch {
          if (cdp) {
            cdp.close()
            cdp = null
          }

          return false
        }
      },
      { timeoutMs: 120000, label: 'isolated renderer + __PERF_DRIVE__' }
    )
    const driverAt = Date.now()

    try {
      await cdp.send('Emulation.setFocusEmulationEnabled', { enabled: true })
    } catch {
      // Older CDP / not supported — fall back to the anti-throttle flags.
    }

    // Renderer-side boot marks (relative to its own navigation start).
    const bootMarks = await readBootMarks(cdp)
    const timings = {
      spawn_to_cdp_ms: cdpAt ? cdpAt - spawnAt : null,
      spawn_to_driver_ms: driverAt - spawnAt,
      ...bootMarks
    }

    let connected = true

    if (!coldStart) {
      // Steady-state scenarios: wait for the gateway to connect (reconnect churn
      // contaminates frame pacing) and let residual cold-start work drain.
      connected = await waitForConnected(cdp, connectTimeoutMs)

      if (!connected) {
        console.warn(
          `[perf] gateway did not connect within ${connectTimeoutMs}ms — ` +
            'stream/frame numbers may be inflated by reconnect churn.'
        )
      }

      await sleep(settleMs)
    }

    return {
      connected,
      cdp,
      devUrl,
      port,
      prod,
      timings,
      teardown: () => {
        cdp?.close()
        teardown()
      }
    }
  } catch (err) {
    teardown()
    throw err
  }
}

// Read First Contentful Paint + time-to-composer from the renderer, relative to
// its navigation start (the process-spawn deltas live in `timings`).
async function readBootMarks(cdp) {
  try {
    return await cdp.eval(`(() => {
      const paints = performance.getEntriesByType('paint')
      const fcp = paints.find(p => p.name === 'first-contentful-paint')
      const composer = document.querySelector('[data-slot="composer-rich-input"]')
      return {
        fcp_ms: fcp ? Math.round(fcp.startTime) : null,
        // performance.now() at read time ≈ time since nav start; only meaningful
        // right after boot (cold-start reads it immediately).
        nav_to_read_ms: Math.round(performance.now()),
        composer_present: !!composer
      }
    })()`)
  } catch {
    return { fcp_ms: null, nav_to_read_ms: null, composer_present: false }
  }
}

export { DESKTOP_DIR }
