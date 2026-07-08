/**
 * backend-probes.cjs
 *
 * Cheap "does this candidate backend actually work" checks used by
 * resolveHermesBackend (main.cjs). The resolver walks a ladder of
 * candidates -- bootstrap marker, `hermes` on PATH, system Python with
 * hermes_cli installed -- and historically returned the first candidate
 * whose binary existed on disk. That assumption breaks when a user has
 * a pre-installed Python 3.11-3.13 (so findSystemPython() returns a
 * path) but no hermes_cli in its site-packages: the resolver hands back
 * a backend the spawn step can't actually run, and the user gets a
 * dead-on-arrival "ModuleNotFoundError: No module named 'hermes_cli'"
 * instead of the first-launch installer.
 *
 * These probes give the resolver a way to verify a candidate before
 * trusting it. Failure (non-zero exit, exception, timeout) means "skip
 * this rung, try the next one"; success means "spawn this for real."
 * Falling off the bottom of the ladder lands on the bootstrap-needed
 * sentinel, which is exactly what we want when nothing pre-existing
 * actually works.
 *
 * Both probes are deliberately fast and forgiving:
 *   - 5s timeout (a hung interpreter beats forever, but we still give
 *     slow disks / cold caches room to breathe)
 *   - stdio ignored (we only care about exit code; stdout/stderr are
 *     not surfaced to the user, just to recentHermesLog for forensics
 *     via the caller's catch block if it chooses)
 *   - any throw -> false (never propagate -- resolver wants a boolean)
 *
 * Kept in a standalone cjs module so it can be unit-tested with
 * `node --test` without dragging in the electron runtime (same pattern
 * as bootstrap-platform.cjs and hardening.cjs).
 */

const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const PROBE_TIMEOUT_MS = 5000

/**
 * Return the Python snippet used to verify Hermes can import far enough to
 * launch the CLI. Kept exported for tests so dependency regressions are
 * caught without needing a real broken venv fixture.
 *
 * @returns {string}
 */
function hermesRuntimeImportProbe() {
  return 'import yaml; import dotenv; import hermes_cli.config'
}

function readPyvenvCfg(venvRoot) {
  if (!venvRoot) return {}
  const cfgPath = path.join(venvRoot, 'pyvenv.cfg')
  try {
    const raw = fs.readFileSync(cfgPath, 'utf8')
    const parsed = {}
    for (const line of raw.split(/\r?\n/)) {
      const idx = line.indexOf('=')
      if (idx === -1) continue
      parsed[line.slice(0, idx).trim().toLowerCase()] = line.slice(idx + 1).trim()
    }
    return parsed
  } catch {
    return {}
  }
}

/**
 * Return true when the venv's recorded base interpreter still exists.
 *
 * Standard CPython venvs record their base interpreter directory in
 * `pyvenv.cfg` as `home = ...`. On Windows uv-managed venv launchers this is
 * load-bearing: `venv\\Scripts\\python.exe` can still exist as a trampoline
 * after the base interpreter under `home` was deleted, at which point any
 * attempt to spawn the venv runtime fails before Python code can run.
 *
 * Missing `pyvenv.cfg` / `home` is treated as "unknown, don't block" so we
 * only force the destructive repair path on a concrete broken-base signal.
 *
 * @param {string} venvRoot
 * @param {object} [opts]
 * @param {boolean} [opts.isWindows]
 * @returns {boolean}
 */
function venvBaseInterpreterPresent(venvRoot, opts = {}) {
  if (!venvRoot) return true
  const cfg = readPyvenvCfg(venvRoot)
  const home = cfg.home
  if (!home) return true
  const pythonName = opts.isWindows ? 'python.exe' : 'python'
  return fs.existsSync(path.join(home, pythonName))
}

/**
 * Return true iff the Hermes runtime import probe exits 0.
 *
 * Used to gate the "fallback to system Python with hermes_cli installed"
 * rung of resolveHermesBackend. Without this, a system Python 3.11-3.13
 * registered in PEP 514 makes findSystemPython() succeed regardless of
 * whether hermes_cli has actually been pip-installed into its
 * site-packages -- and the resolver returns a backend that immediately
 * dies on spawn.
 *
 * The probe intentionally imports hermes_cli.config, not just the top-level
 * package: a broken/empty Windows launcher venv can still see the source tree
 * through PYTHONPATH but lack PyYAML, then die on the first real CLI import.
 *
 * @param {string} pythonPath - Absolute path to a python.exe / python.
 * @param {object} [opts]
 * @param {object} [opts.env] - Additional environment for the probe.
 * @returns {boolean}
 */
function canImportHermesCli(pythonPath, opts = {}) {
  if (!pythonPath) return false
  try {
    execFileSync(pythonPath, ['-c', hermesRuntimeImportProbe()], {
      env: { ...process.env, ...(opts.env || {}) },
      stdio: 'ignore',
      timeout: PROBE_TIMEOUT_MS,
      windowsHide: true
    })
    return true
  } catch {
    return false
  }
}

/**
 * Return true iff `<hermesCommand> --version` exits 0.
 *
 * Used to gate the "existing `hermes` on PATH" rung. Without this, a
 * stale hermes.cmd shim left behind by an uninstalled pip install (or
 * a half-built venv whose `hermes` entry-point points at a deleted
 * Python) survives findOnPath() and gets selected as the backend.
 *
 * We intentionally avoid invoking the command with the dashboard args
 * here -- `--version` is the cheapest "is this binary alive" smoke
 * test that every hermes_cli entry-point has supported since 0.1.
 *
 * @param {string} hermesCommand - Resolved absolute path to a hermes
 *   executable (or an interpreter+script wrapper).
 * @param {object} [opts]
 * @param {boolean} [opts.shell] - Whether to run through a shell. For
 *   .cmd/.bat shims on Windows execFileSync needs shell:true to find
 *   the cmd interpreter; mirrors the same flag isCommandScript() drives
 *   in resolveHermesBackend.
 * @returns {boolean}
 */
function verifyHermesCli(hermesCommand, opts = {}) {
  if (!hermesCommand) return false
  try {
    execFileSync(hermesCommand, ['--version'], {
      stdio: 'ignore',
      timeout: PROBE_TIMEOUT_MS,
      shell: Boolean(opts.shell),
      windowsHide: true
    })
    return true
  } catch {
    return false
  }
}

module.exports = {
  canImportHermesCli,
  hermesRuntimeImportProbe,
  readPyvenvCfg,
  verifyHermesCli,
  PROBE_TIMEOUT_MS,
  venvBaseInterpreterPresent
}
