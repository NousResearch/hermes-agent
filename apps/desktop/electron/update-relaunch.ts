/**
 * update-relaunch.ts — pure decision + script-generation helpers for the
 * Linux in-app update relaunch (#45205).
 *
 * Extracted from main.ts's `applyUpdatesPosixInApp` so the security- and
 * correctness-critical "do we relaunch, or land on a manual terminal state?"
 * decision is unit-testable without booting Electron (main.ts
 * `require('electron')` at load).
 *
 * Background
 * ----------
 * After `hermes update` + `hermes desktop --build-only`, the freshly-rebuilt
 * GUI lives under `apps/desktop/release/<plat>-unpacked`. We can only honestly
 * relaunch into the new GUI when the *running* binary is that rebuilt one —
 * i.e. its execPath is under the rebuilt `release/<plat>-unpacked` dir.
 *
 *   - Source / unpacked install (execPath under release/<plat>-unpacked):
 *     the running binary IS the thing we just rebuilt → relaunch it in place.
 *   - AppImage / .deb / .rpm / dev / unresolved (execPath elsewhere):
 *     the backend was updated but THIS GUI shell was NOT replaced. Claiming
 *     "the new version loads next launch" is a lie that produces GUI/backend
 *     skew (#37541): the user keeps running the old GUI against new backend
 *     code with no path to fix it from inside the app. Surface an explicit
 *     terminal state telling them the GUI package must be reinstalled.
 *
 * Sandbox preflight (#3 in the review)
 * ------------------------------------
 * A fresh `release/<plat>-unpacked` rebuild can leave `chrome-sandbox` without
 * the required `root:root` + setuid (mode 4755). Electron then refuses to
 * launch with "The SUID sandbox helper binary was found, but is not configured
 * correctly" and the relaunch yields "quit and never came back" — a dead app.
 * Before we quit+hand off we preflight the rebuilt sandbox helper; if it is NOT
 * launchable (and no working non-interactive fallback applies — see
 * sandboxFallbackFromEnv) we DO NOT quit. We keep the working window and return
 * the closeable manual-restart terminal state instead.
 */

import path from 'node:path'

function pathForPlatform(platform) {
  return platform === 'win32' ? path.win32 : path.posix
}

// Map process.platform → electron-builder's `release/<dir>-unpacked` name.
function unpackedDirName(platform) {
  if (platform === 'darwin') {
    return 'mac-unpacked'
  } // not used (mac swaps bundles)

  if (platform === 'win32') {
    return 'win-unpacked'
  }

  return 'linux-unpacked'
}

/**
 * If `execPath` lives under `<updateRoot>/apps/desktop/release/<plat>-unpacked`,
 * return that unpacked dir; otherwise null. A null result means the running
 * binary is NOT the thing we just rebuilt (AppImage/.deb/.rpm/dev), so we must
 * not claim a GUI relaunch.
 *
 * Match is a path-segment-aware prefix check (not a bare string startsWith) so
 * `.../release/linux-unpacked-evil` can't masquerade as `.../release/linux-unpacked`.
 */
function resolveUnpackedRelease(execPath, updateRoot, platform) {
  if (!execPath || !updateRoot) {
    return null
  }

  const platformPath = pathForPlatform(platform)
  const releaseDir = platformPath.join(updateRoot, 'apps', 'desktop', 'release')
  const unpacked = platformPath.join(releaseDir, unpackedDirName(platform))
  const normalizedExec = platformPath.resolve(String(execPath))
  // execPath must be the unpacked dir itself or a descendant of it.
  const withSep = unpacked.endsWith(platformPath.sep) ? unpacked : unpacked + platformPath.sep

  if (normalizedExec === unpacked || normalizedExec.startsWith(withSep)) {
    return unpacked
  }

  return null
}

/**
 * Pure decision: given whether the running binary is under the rebuilt
 * unpacked release AND whether its sandbox helper is launchable, choose the
 * terminal outcome.
 *
 *   'relaunch' — quit + detached watcher re-execs the rebuilt binary in place.
 *   'guiSkew'  — backend updated, GUI package NOT changed; user must reinstall
 *                the GUI. Closeable terminal state; does NOT claim a GUI update.
 *   'manual'   — running the rebuilt binary, but its sandbox helper is not
 *                launchable and no fallback applies; do NOT quit into a dead
 *                app. Closeable manual-restart terminal state.
 */
function decideRelaunchOutcome({ underUnpacked, sandboxOk }) {
  if (!underUnpacked) {
    return 'guiSkew'
  }

  if (!sandboxOk) {
    return 'manual'
  }

  return 'relaunch'
}

/**
 * Preflight the rebuilt sandbox helper. Returns
 *   { ok: boolean, reason: string, path: string }
 *
 * `ok` is true when chrome-sandbox is owned by uid 0 AND has the setuid bit
 * (mode & 0o4000) — i.e. Electron can launch it. If chrome-sandbox does not
 * exist at all we treat it as ok: this Electron build does not use the SUID
 * sandbox helper (e.g. it ships the namespace sandbox), so the relaunch is not
 * blocked on it.
 *
 * `statSync` is injectable so this is testable without a real setuid file.
 */
function sandboxPreflight(unpackedDir, statSync) {
  if (!unpackedDir) {
    return { ok: false, reason: 'no-unpacked-dir', path: null }
  }

  const sandboxPath = path.join(unpackedDir, 'chrome-sandbox')
  let st

  try {
    st = statSync(sandboxPath)
  } catch {
    // No chrome-sandbox helper present → this build doesn't rely on the SUID
    // sandbox; nothing to block the relaunch.
    return { ok: true, reason: 'no-sandbox-helper', path: sandboxPath }
  }

  const ownedByRoot = st.uid === 0
  const hasSetuid = (st.mode & 0o4000) !== 0

  if (ownedByRoot && hasSetuid) {
    return { ok: true, reason: 'launchable', path: sandboxPath }
  }

  if (!ownedByRoot && !hasSetuid) {
    return { ok: false, reason: 'not-root-not-setuid', path: sandboxPath }
  }

  if (!ownedByRoot) {
    return { ok: false, reason: 'not-root', path: sandboxPath }
  }

  return { ok: false, reason: 'not-setuid', path: sandboxPath }
}

/**
 * Detect a non-interactive sandbox fallback the user has opted into via the
 * environment. The reviewer asked us to integrate with any existing
 * `--no-sandbox` / chrome-sandbox handling. A repo grep found NO existing
 * non-interactive sandbox fallback in the desktop app (the only chrome-sandbox
 * reference is documentation in scripts/before-pack.ts). The one signal that
 * DOES exist is the standard Electron escape hatch: ELECTRON_DISABLE_SANDBOX=1
 * (and the equivalent `--no-sandbox` already present in the launch args). If
 * the user has set that, the rebuilt binary will start even with a broken
 * chrome-sandbox, so the relaunch is safe.
 *
 * Returns true when a fallback makes the relaunch safe despite a failed
 * sandbox preflight.
 */
function sandboxFallbackFromEnv(env, launchArgs) {
  const disable = String((env && env.ELECTRON_DISABLE_SANDBOX) || '').trim()

  if (disable === '1' || disable.toLowerCase() === 'true') {
    return true
  }

  if (Array.isArray(launchArgs) && launchArgs.some(a => a === '--no-sandbox')) {
    return true
  }

  return false
}

// POSIX single-quote a value for safe inclusion in the generated bash script.
function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

// Electron / Chromium internal switches that must NOT be replayed on re-exec:
// they are runtime artifacts of THIS launch, not user intent, and re-passing
// them can change sandbox/zygote behavior or point at stale fds/dirs.
const INTERNAL_ARG_PREFIXES = [
  '--type=', // renderer/gpu/zygote child markers
  '--user-data-dir=',
  '--enable-features=',
  '--disable-features=',
  '--field-trial-handle=',
  '--enable-logging',
  '--log-file=',
  // NB: --no-sandbox is deliberately NOT stripped — it reflects the user's /
  // environment's SUID-sandbox opt-out (some hardened kernels/containers require
  // it) and is the signal sandboxFallbackFromEnv() uses to allow a relaunch when
  // chrome-sandbox isn't setuid. Dropping it would make exactly that relaunch
  // fail ("quit and never came back").
  '--disable-gpu-sandbox',
  '--lang=',
  '--inspect',
  '--remote-debugging-port='
]

/**
 * Filter Electron internals out of the original launch args so we replay only
 * meaningful user/launcher intent (deep-link URLs, app-specific flags).
 * `argv` is expected to be process.argv.slice(1) for a PACKAGED app (argv[0] is
 * the exec path itself; there is no entry-script arg as in a dev run).
 */
function collectRelaunchArgs(argv) {
  if (!Array.isArray(argv)) {
    return []
  }

  return argv.filter(arg => {
    if (typeof arg !== 'string' || arg.length === 0) {
      return false
    }

    return !INTERNAL_ARG_PREFIXES.some(prefix =>
      prefix.endsWith('=') ? arg.startsWith(prefix) : arg === prefix || arg.startsWith(prefix + '=')
    )
  })
}

// Env keys whose values define the relaunched instance's context (which
// backend/profile/root it talks to). Anything HERMES_DESKTOP_* is preserved
// plus HERMES_HOME. We snapshot the values, not the live env, so the new
// instance comes up pointed at the same place this one was.
// ELECTRON_DISABLE_SANDBOX is preserved for the same reason --no-sandbox is kept
// in the replayed args: if a relaunch is only safe because the user opted out of
// the SUID sandbox, the relaunched instance must inherit that opt-out too.
const PRESERVED_ENV_KEYS = ['HERMES_HOME', 'ELECTRON_DISABLE_SANDBOX']
const PRESERVED_ENV_PREFIXES = ['HERMES_DESKTOP_']

function collectRelaunchEnv(env) {
  const out = {}

  if (!env || typeof env !== 'object') {
    return out
  }

  for (const [key, value] of Object.entries(env)) {
    if (value == null) {
      continue
    }

    if (PRESERVED_ENV_KEYS.includes(key) || PRESERVED_ENV_PREFIXES.some(p => key.startsWith(p))) {
      out[key] = String(value)
    }
  }

  return out
}

/**
 * Build the detached bash watcher that waits for the parent to exit (graceful
 * window then SIGKILL), self-deletes, and re-execs the rebuilt binary WITH the
 * original launch context (cwd, env, args) restored.
 *
 * @param {object} o
 * @param {number} o.pid       parent (this) process pid to wait on
 * @param {string} o.execPath  binary to re-exec
 * @param {string[]} o.args    filtered launch args to replay
 * @param {object} o.env       env key→value to export before exec
 * @param {string} o.cwd       working directory to restore
 */
function buildRelaunchScript({ pid, execPath, args, env, cwd }) {
  const exports = Object.entries(env || {})
    .map(([k, v]) => `export ${k}=${shellQuote(v)}`)
    .join('\n')

  const quotedArgs = (args || []).map(shellQuote).join(' ')
  const cwdLine = cwd ? `cd ${shellQuote(cwd)} 2>/dev/null || true` : ''

  // NOTE: `exec` replaces the watcher process with the relaunched app, so the
  // re-exec inherits exactly the env/cwd we set above.
  return `#!/bin/bash
set -u
APP_PID=${Number(pid)}
# Wait up to ~30s for a graceful exit, then SIGKILL: a hung/zombie parent must
# be gone before we relaunch, or the new instance bails on the single-instance
# lock. (#45205)
for _ in $(seq 1 60); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "$APP_PID" 2>/dev/null; then
  kill -9 "$APP_PID" 2>/dev/null || true
  sleep 0.5
fi
# Self-delete so temp watchers don't accumulate across updates.
rm -f -- "$0" 2>/dev/null || true
${cwdLine}
${exports}
exec ${shellQuote(execPath)}${quotedArgs ? ' ' + quotedArgs : ''}
`
}

export {
  buildRelaunchScript,
  collectRelaunchArgs,
  collectRelaunchEnv,
  decideRelaunchOutcome,
  INTERNAL_ARG_PREFIXES,
  PRESERVED_ENV_KEYS,
  PRESERVED_ENV_PREFIXES,
  resolveUnpackedRelease,
  sandboxFallbackFromEnv,
  sandboxPreflight,
  shellQuote,
  unpackedDirName
}
