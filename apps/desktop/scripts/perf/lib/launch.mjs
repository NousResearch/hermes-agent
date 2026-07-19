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

function runNode(scriptRelPath, args = []) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(process.execPath, [join(DESKTOP_DIR, scriptRelPath), ...args], {
      cwd: DESKTOP_DIR,
      stdio: 'inherit'
    })
    child.on('error', reject)
    child.on('exit', code => (code === 0 ? resolveRun() : reject(new Error(`${scriptRelPath} exited ${code}`))))
  })
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
export async function startIsolatedInstance({
  port = 9222,
  devPort = 5174,
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
  const devUrl = `http://127.0.0.1:${devPort}`

  // Only seed a temp home we created — never scribble into a user-provided one.
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
    // 1. Renderer: reuse an already-running dev server, else start one.
    if (!(await reachable(devUrl))) {
      const viteBin = resolveViteBin()
      const vite = spawn(process.execPath, [viteBin, '--host', '127.0.0.1', '--port', String(devPort)], {
        cwd: DESKTOP_DIR,
        stdio: ['ignore', 'inherit', 'inherit']
      })
      children.push(vite)
      await waitFor(() => reachable(devUrl), { timeoutMs: 60000, label: `vite dev server on :${devPort}` })
    }

    // 2. Electron main bundle (dev variant) — same step the dev script runs.
    await runNode('scripts/bundle-electron-main.mjs', ['--dev'])

    // 3. Isolated Electron. --user-data-dir gives it its own single-instance
    //    lock scope; HERMES_HOME gives it its own backend + sessions.
    const electronBin = require('electron')
    const electron = spawn(
      electronBin,
      [
        '.',
        `--user-data-dir=${userData}`,
        `--remote-debugging-port=${port}`,
        // The perf window usually opens behind the user's other windows, and
        // Chromium throttles frame production for backgrounded/occluded windows
        // (~17fps), which shows up as choppy frames with ZERO longtasks and
        // wrecks the stream frame-pacing metric. Disable every throttle path so
        // measurements reflect real render cost regardless of window state
        // (CalculateNativeWinOcclusion is the macOS/Windows occlusion detector).
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--disable-features=CalculateNativeWinOcclusion'
      ],
      {
        cwd: DESKTOP_DIR,
        stdio: ['ignore', 'inherit', 'inherit'],
        env: {
          ...process.env,
          HERMES_HOME: home,
          HERMES_DESKTOP_DEV_SERVER: devUrl,
          HERMES_DESKTOP_BOOT_FAKE: '1',
          HERMES_DESKTOP_BOOT_FAKE_STEP_MS: String(bootFakeStepMs),
          XCURSOR_SIZE: '24'
        }
      }
    )
    children.push(electron)

    // 4. Wait for the renderer + the perf driver to be live.
    let cdp = null
    await waitFor(
      async () => {
        try {
          cdp = await CDP.connect({ port, match: String(devPort), timeoutMs: 2000 })

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

    // Electron throttles rAF/timers for a window that isn't foregrounded
    // (per-window backgroundThrottling, which the Chromium CLI flags above don't
    // override). Focus emulation makes the renderer behave as if focused so
    // frame-pacing measurements are real even though the perf window sits behind
    // the user's other windows — WITHOUT actually stealing OS focus.
    try {
      // Behave as if focused so frame-pacing isn't throttled while the perf
      // window sits behind the user's IDE/terminal — WITHOUT stealing OS focus.
      await cdp.send('Emulation.setFocusEmulationEnabled', { enabled: true })
    } catch {
      // Older CDP / not supported — fall back to the anti-throttle flags above.
    }

    // Wait for the gateway socket to actually open. A booting/absent backend
    // retries on a 1–15s backoff, and that churn contaminates frame-pacing
    // (the `stream` scenario). Best-effort: proceed after the timeout so the
    // backend-independent scenarios (keystroke, transcript) still run.
    const connected = await waitForConnected(cdp, connectTimeoutMs)

    if (!connected) {
      console.warn(
        `[perf] gateway did not connect within ${connectTimeoutMs}ms — ` +
          'stream/frame numbers may be inflated by reconnect churn.'
      )
    }

    // Let residual cold-start work (vite dep pre-bundling, initial paint) drain.
    await sleep(settleMs)

    return {
      connected,
      cdp,
      devUrl,
      port,
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

export { DESKTOP_DIR }
