/**
 * backend-probes.ts
 *
 * Cheap "does this candidate backend actually work" checks used by
 * resolveHermesBackend (main.ts). The resolver walks a ladder of
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
 *   - default 15s timeout (5s was too short on cold Windows disks / AV;
 *     issue #61764 death-loop) with HERMES_PROBE_TIMEOUT_MS override
 *   - one automatic retry after a timeout before declaring the runtime dead
 *   - stdio ignored (we only care about exit code; stdout/stderr are
 *     not surfaced to the user, just to recentHermesLog for forensics
 *     via the caller's catch block if it chooses)
 *   - any throw -> false (never propagate -- resolver wants a boolean)
 *
 * Kept in a standalone ts module so it can be unit-tested with
 * `node --test` without dragging in the electron runtime (same pattern
 * as bootstrap-platform.ts and hardening.ts).
 */

import { execFileSync } from 'node:child_process'
import path from 'node:path'

/**
 * Python versions Hermes supports, oldest first.
 *
 * Single source of truth for the floor `pyproject.toml` declares
 * (`requires-python = ">=3.11,<3.14"`). main.ts's Windows detection passes
 * (PEP 514 registry, standard install dirs, `py.exe -<version>`) and the POSIX
 * candidate walk below both derive from this list, so the floor can't drift
 * between platforms.
 */
const SUPPORTED_PYTHON_VERSIONS = ['3.11', '3.12', '3.13'] as const

/**
 * Ordered, bounded list of POSIX interpreter command names to try.
 *
 * Explicitly-versioned names come first so a box whose bare `python3` is out
 * of range (Debian 11 / Ubuntu 20.04 ship 3.9; macOS CommandLineTools ships
 * 3.9.6) still finds a supported interpreter instead of stopping at the first
 * PATH hit. Bare `python3` / `python` stay as lower rungs rather than being
 * removed: a venv or conda env on PATH exposes only the bare name, and the
 * caller's probe -- not the file name -- is what proves a candidate runnable.
 *
 * Windows is deliberately excluded: bare PATH lookup there hits the Microsoft
 * Store redirector, so main.ts uses registry / install-dir / `py.exe` passes.
 *
 * @returns {string[]} Command names in priority order.
 */
function posixPythonCommandCandidates() {
  return [...SUPPORTED_PYTHON_VERSIONS.map(version => `python${version}`), 'python3', 'python']
}

/**
 * Directories searched for a versioned interpreter BEFORE PATH, on macOS only.
 *
 * A Finder/Dock launch does not run a login shell, so the app inherits
 * launchd's environment -- `launchctl getenv PATH` is unset on a stock machine,
 * which leaves `/usr/bin:/bin:/usr/sbin:/sbin`. Homebrew is not on it. That is
 * the same inheritance problem `backend-env.POSIX_SANE_PATH_ENTRIES` exists to
 * repair for the spawned backend's PATH, and the same one main.ts's `gh` lookup
 * dodges by hard-coding these two directories -- but interpreter resolution
 * still ran PATH-only, so on a GUI launch it could not see a perfectly good
 * Homebrew 3.12 and fell through to `/usr/bin/python3` (3.9.6), the interpreter
 * that cannot import hermes_cli at all.
 *
 * Both Homebrew prefixes are listed: `/opt/homebrew` on Apple Silicon,
 * `/usr/local` on Intel (and where the python.org installer symlinks). Linux is
 * excluded on purpose -- its desktop launchers inherit a session PATH built
 * from the user's profile, so PATH already sees a versioned interpreter.
 */
const MACOS_WELL_KNOWN_PYTHON_DIRS = Object.freeze(['/opt/homebrew/bin', '/usr/local/bin'])

/**
 * Versioned interpreter paths in the well-known macOS directories.
 *
 * Versioned names ONLY. Bare `python3` in these directories is a Homebrew
 * symlink to whichever formula is currently linked -- 3.14 on this machine,
 * which is above the `<3.14` ceiling -- so preferring it over PATH would demote
 * a good PATH interpreter in favour of a guess. `python3.12` names its own
 * version, and bare `python3` remains reachable through the PATH walk below.
 *
 * Bounded by construction: `SUPPORTED_PYTHON_VERSIONS.length` x
 * `MACOS_WELL_KNOWN_PYTHON_DIRS.length` paths, each an lstat, and only ones
 * that exist ever become probe candidates.
 *
 * @param {boolean} isMacOS - Non-darwin platforms get an empty list.
 * @returns {string[]} Absolute candidate paths, highest version last-resort
 *   ordering matching SUPPORTED_PYTHON_VERSIONS (oldest first, as elsewhere).
 */
function macOSWellKnownPythonCandidates(isMacOS: boolean) {
  if (!isMacOS) {
    return []
  }

  return MACOS_WELL_KNOWN_PYTHON_DIRS.flatMap(directory =>
    SUPPORTED_PYTHON_VERSIONS.map(version => path.posix.join(directory, `python${version}`))
  )
}

/**
 * Walk interpreter candidates and return the first one `accept` approves.
 *
 * Pure and dependency-injected (no electron, no direct PATH access) so the
 * ordering policy is unit-testable without a real interpreter zoo on disk.
 * Resolved paths are de-duplicated, so the common case where several names
 * point at the same binary (`python3` and `python` symlinked together) costs
 * one probe, not two.
 *
 * Two phases, in order:
 *   1. versioned interpreters in the well-known macOS directories (only when
 *      `fileExists` and a darwin `platform` are both supplied)
 *   2. the PATH walk, versioned names before bare ones
 *
 * Phase 1 is a PREFERENCE, structurally identical to the versioned-before-bare
 * ordering in phase 2: `accept` still decides. A Homebrew interpreter that
 * fails the probe is demoted and the walk continues into PATH, so the scan can
 * only ever change WHICH acceptable interpreter wins -- never whether an
 * unacceptable one gets returned.
 *
 * @param {object} deps
 * @param {(command: string) => string | null} deps.findOnPath - PATH resolver.
 * @param {(candidate: string) => boolean} [deps.accept] - Validator; when
 *   omitted the first resolvable candidate wins (legacy behaviour).
 * @param {(filePath: string) => boolean} [deps.fileExists] - Enables phase 1.
 *   Omitted (the default) means PATH-only, which keeps callers that don't want
 *   to touch the filesystem -- and tests that inject no fake fs -- hermetic.
 * @param {string} [deps.platform] - Must be 'darwin' for phase 1 to run.
 * @param {string[]} [deps.candidates] - Override the command list (tests).
 * @returns {string | null} An accepted interpreter path, or null.
 */
function findAcceptablePython(deps: {
  findOnPath: (command: string) => string | null | undefined
  accept?: ((candidate: string) => boolean) | null
  fileExists?: ((filePath: string) => boolean) | null
  platform?: string | null
  candidates?: string[]
}) {
  const candidates = deps.candidates || posixPythonCommandCandidates()
  const seen = new Set<string>()

  const consider = (resolved: string | null | undefined) => {
    if (!resolved || seen.has(resolved)) {
      return null
    }

    seen.add(resolved)

    return !deps.accept || deps.accept(resolved) ? resolved : null
  }

  if (deps.fileExists && deps.platform === 'darwin') {
    for (const candidate of macOSWellKnownPythonCandidates(true)) {
      if (!deps.fileExists(candidate)) {
        continue
      }

      const accepted = consider(candidate)

      if (accepted) {
        return accepted
      }
    }
  }

  for (const command of candidates) {
    const accepted = consider(deps.findOnPath(command))

    if (accepted) {
      return accepted
    }
  }

  return null
}

/**
 * Actionable message for "no interpreter on this machine can run Hermes."
 *
 * The failure this replaces was actively misleading: an out-of-range
 * interpreter got spawned anyway and died inside hermes_cli with a bare
 * `TypeError: unsupported operand type(s) for |` from PEP 604 syntax, which
 * invites the wrong fix (making a `>=3.11` codebase 3.9-compatible) instead of
 * pointing at interpreter selection.
 *
 * @param {string} root - Hermes source root the interpreter was resolved for.
 * @returns {string} Log-ready message naming both remedies.
 */
function pythonResolutionFailureMessage(root: string) {
  const floor = SUPPORTED_PYTHON_VERSIONS[0]
  const venvHint = root ? `${root}/.venv` : '<hermes root>/.venv'

  return (
    `No Python >= ${floor} with Hermes dependencies found — expected a venv at ${venvHint}, ` +
    'or set HERMES_DESKTOP_PYTHON to an interpreter that has them.'
  )
}

/**
 * In-checkout venv interpreter paths for `root`, in precedence order.
 *
 * `.venv` first (what `uv sync` creates, and what CI's `uv sync --locked
 * --python 3.11` produces), then `venv` (what `scripts/install.sh` creates when
 * it isn't redirecting via UV_PROJECT_ENVIRONMENT).
 *
 * @param {string} root - Hermes source root.
 * @param {boolean} isWindows - Selects Scripts/python.exe vs bin/python.
 * @returns {string[]} Absolute candidate paths, highest precedence first.
 */
function venvPythonCandidates(root: string, isWindows: boolean) {
  const relativePaths = isWindows
    ? [path.join('.venv', 'Scripts', 'python.exe'), path.join('venv', 'Scripts', 'python.exe')]
    : [path.join('.venv', 'bin', 'python'), path.join('venv', 'bin', 'python')]

  return relativePaths.map(relativePath => path.join(root, relativePath))
}

/**
 * The whole interpreter-precedence ladder for a Hermes source root, as one
 * pure function. Extracted here (rather than left inline in main.ts) so the
 * precedence policy is written down once and unit-testable without electron:
 *
 *   1. explicit `HERMES_DESKTOP_PYTHON` override, when it exists on disk
 *   2. `<root>/.venv`, then `<root>/venv`
 *   3. a system interpreter that passes `accept`
 *
 * Rungs 1-2 are trusted unprobed on purpose. An override is a deployment
 * contract (the `hgui` worktree helper points it at a shared venv), and a venv
 * inside the checkout is the layout every install path produces -- probing
 * either would spend a subprocess to second-guess a deliberate choice, and a
 * broken venv there is a repair case, not a "silently use something else" case.
 * Rung 3 is the only rung that picks an interpreter the user never pointed at,
 * which is exactly why it is the one that must be proven runnable.
 *
 * The rung-3 gate is built HERE, not by the caller: `canImport` is a required
 * dependency, so the walk cannot be wired up without it. That is deliberate --
 * an optional gate is one an edit can quietly drop, which is how this rung came
 * to be the only one in the ladder running unprobed.
 *
 * @param {object} deps
 * @param {string} deps.root - Hermes source root.
 * @param {string} [deps.override] - HERMES_DESKTOP_PYTHON value.
 * @param {boolean} deps.isWindows
 * @param {(filePath: string) => boolean} deps.fileExists
 * @param {(candidate: string, root: string) => boolean} deps.canImport - Proof
 *   that a candidate can import Hermes with `root` on PYTHONPATH.
 * @param {(accept: (candidate: string) => boolean) => string | null}
 *   deps.findSystemPython - Platform walk; must honour the accept validator.
 * @returns {string | null} Interpreter path, or null when nothing qualifies.
 */
function resolvePythonForRoot(deps: {
  root: string
  override?: string | null
  isWindows: boolean
  fileExists: (filePath: string) => boolean
  canImport: (candidate: string, root: string) => boolean
  findSystemPython: (accept: (candidate: string) => boolean) => string | null | undefined
}) {
  if (deps.override && deps.fileExists(deps.override)) {
    return deps.override
  }

  for (const candidate of venvPythonCandidates(deps.root, deps.isWindows)) {
    if (deps.fileExists(candidate)) {
      return candidate
    }
  }

  return deps.findSystemPython(candidate => deps.canImport(candidate, deps.root)) || null
}

/** Default probe budget. 5s false-negativeed healthy Windows cold starts (#61764). */
const DEFAULT_PROBE_TIMEOUT_MS = 15_000

/**
 * Resolve the backend probe timeout (ms).
 * Honours HERMES_PROBE_TIMEOUT_MS when it parses as a positive integer.
 */
function resolveProbeTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.HERMES_PROBE_TIMEOUT_MS

  if (raw == null || raw === '') {
    return DEFAULT_PROBE_TIMEOUT_MS
  }

  const n = Number.parseInt(String(raw), 10)

  if (!Number.isFinite(n) || n <= 0) {
    return DEFAULT_PROBE_TIMEOUT_MS
  }

  // Clamp absurd values (ms) so a typo can't hang startup forever.
  return Math.min(n, 120_000)
}

const PROBE_TIMEOUT_MS = resolveProbeTimeoutMs()

function isTimeoutError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false
  }

  const e = err as { code?: string; killed?: boolean; signal?: string }

  if (e.killed === true) {
    return true
  }

  if (e.code === 'ETIMEDOUT') {
    return true
  }

  // Node marks timed-out execFileSync with SIGTERM on some platforms.
  if (e.signal === 'SIGTERM') {
    return true
  }

  return false
}

/**
 * Run execFileSync; on timeout only, retry once before failing.
 * Non-timeout failures (ENOENT, non-zero exit) fail immediately.
 */
function execProbeSync(
  command: string,
  args: string[],
  options: {
    cwd?: string
    env?: NodeJS.ProcessEnv
    stdio: 'ignore'
    timeout: number
    shell?: boolean
    windowsHide?: boolean
  }
): void {
  try {
    execFileSync(command, args, options)
  } catch (err) {
    if (!isTimeoutError(err)) {
      throw err
    }

    // One cold-cache / AV miss should not force hermes-setup --update (#61764).
    execFileSync(command, args, options)
  }
}

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
 * @param {object} [opts.env] - Additional environment for the probe.
 * @returns {boolean}
 */
function canImportHermesCli(pythonPath: string, opts: { env?: Record<string, string> } = {}) {
  if (!pythonPath) {
    return false
  }

  try {
    execProbeSync(pythonPath, ['-c', hermesRuntimeImportProbe()], {
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
 * @param {boolean} [opts.shell] - Whether to run through a shell. For
 *   .cmd/.bat shims on Windows execFileSync needs shell:true to find
 *   the cmd interpreter; mirrors the same flag isCommandScript() drives
 *   in resolveHermesBackend.
 * @returns {boolean}
 */
/**
 * An explicit desktop backend command is a deployment contract, not a PATH
 * discovery candidate. In particular, the Nix desktop wrapper points this at
 * its immutable, matching Hermes package; it must never fall through to the
 * mutable install-script bootstrap path if a best-effort probe is slow.
 */
function shouldTrustHermesOverride(hermesOverride?: string) {
  return typeof hermesOverride === 'string' && hermesOverride.trim().length > 0
}

function verifyHermesCli(hermesCommand: string, opts?: { shell?: boolean }) {
  if (!hermesCommand) {
    return false
  }

  try {
    execProbeSync(hermesCommand, ['--version'], {
      stdio: 'ignore',
      timeout: PROBE_TIMEOUT_MS,
      shell: Boolean(opts?.shell),
      windowsHide: true
    })

    return true
  } catch {
    return false
  }
}

export {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
  execProbeSync,
  findAcceptablePython,
  hermesRuntimeImportProbe,
  MACOS_WELL_KNOWN_PYTHON_DIRS,
  macOSWellKnownPythonCandidates,
  posixPythonCommandCandidates,
  PROBE_TIMEOUT_MS,
  pythonResolutionFailureMessage,
  resolveProbeTimeoutMs,
  resolvePythonForRoot,
  shouldTrustHermesOverride,
  SUPPORTED_PYTHON_VERSIONS,
  venvPythonCandidates,
  verifyHermesCli
}
