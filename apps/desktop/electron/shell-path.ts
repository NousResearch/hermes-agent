import { execFile } from 'node:child_process'

import { appendUniquePathEntries, delimiterForPlatform, pathEnvKey } from './backend-env'

// Login-shell PATH resolution for GUI launches.
//
// A desktop app launched from Finder/Dock (macOS launchd) or a desktop
// environment's launcher (Linux) inherits a minimal PATH such as
// /usr/bin:/bin:/usr/sbin:/sbin — none of the user's shell profiles ever run.
// backend-env.ts papers over the common cases with a static sane-entry list
// (Homebrew, /usr/local), but anything only a profile adds — ~/.local/bin,
// nvm/pyenv/asdf shims, ~/.cargo/bin, nix profiles — stays invisible to the
// backend process. That breaks tool availability checks (shutil.which), stdio
// MCP server spawns, and the Electron-side binary resolvers.
//
// Fix (same approach as VS Code's shell-environment resolution; ported from
// cline/cline#12429): run the user's shell once as an interactive login shell,
// capture $PATH between sentinel markers so profile banners can't corrupt the
// value, and merge it into process.env — login-shell entries first (so
// /opt/homebrew/bin wins), current-only entries appended (so dirs injected by
// the launching environment aren't lost).
//
// Failure hardening: a broken/slow shell profile must never brick app startup.
// Every attempt is bounded by a timeout, stdin is closed immediately, and any
// failure leaves PATH untouched.

const PATH_START = '__HERMES_LOGIN_PATH_START__'
const PATH_END = '__HERMES_LOGIN_PATH_END__'
const PROBE_COMMAND = "printf '%s' \"" + PATH_START + '${PATH}' + PATH_END + '"'
const ATTEMPT_TIMEOUT_MS = 5000

function loginShellExecutable(env: any = process.env, platform = process.platform) {
  const shell = typeof env?.SHELL === 'string' ? env.SHELL.trim() : ''

  if (shell) {
    return shell
  }

  // macOS Catalina+ defaults to zsh; most Linux distros default to bash.
  return platform === 'darwin' ? '/bin/zsh' : '/bin/bash'
}

// Extract $PATH from between the sentinel markers. Uses the LAST start marker
// so a profile that echoes the environment (or the command line itself) can't
// poison the capture with an earlier partial match.
function extractSentinelPath(stdout) {
  const text = String(stdout || '')
  const start = text.lastIndexOf(PATH_START)

  if (start === -1) {
    return null
  }

  const valueStart = start + PATH_START.length
  const end = text.indexOf(PATH_END, valueStart)

  if (end === -1) {
    return null
  }

  return text.slice(valueStart, end).trim() || null
}

// Login-shell entries first (Homebrew/version-manager dirs win), then any
// current-only entries appended, duplicates and empties dropped.
function mergeLoginShellPath(loginPath, currentPath, { delimiter = ':' }: any = {}) {
  return appendUniquePathEntries([loginPath, currentPath], { delimiter })
}

function runProbe(shell, flags, execFileFn, timeoutMs): Promise<string | null> {
  return new Promise(resolve => {
    let settled = false

    const finish = value => {
      if (!settled) {
        settled = true
        resolve(value)
      }
    }

    try {
      const child = execFileFn(
        shell,
        [...flags, PROBE_COMMAND],
        { encoding: 'utf8', timeout: timeoutMs, windowsHide: true },
        (_error, stdout) => {
          // A profile script may exit nonzero after the sentinel already
          // printed — trust the sentinel, not the exit code.
          finish(extractSentinelPath(stdout))
        }
      )

      // Interactive shells with a broken rc can block reading stdin.
      child?.stdin?.end?.()
    } catch {
      finish(null)
    }
  })
}

async function captureLoginShellPath({
  env = process.env,
  platform = process.platform,
  execFileFn = execFile,
  timeoutMs = ATTEMPT_TIMEOUT_MS
}: any = {}) {
  if (platform === 'win32') {
    // GUI apps on Windows inherit the user PATH from the registry env block;
    // the launchd-minimal-PATH problem is POSIX-only.
    return null
  }

  const shell = loginShellExecutable(env, platform)

  // -l sources ~/.zprofile / ~/.profile (where `brew shellenv` lives); -i
  // sources ~/.zshrc / ~/.bashrc (where nvm/pyenv-style managers live). Some
  // shells swallow combined -ilc with a non-tty stdin (macOS system bash 3.2
  // — see tests/tools/test_find_shell.py), so fall back to a plain login
  // shell before giving up.
  for (const flags of [['-ilc'], ['-lc']]) {
    const captured = await runProbe(shell, flags, execFileFn, timeoutMs)

    if (captured) {
      return captured
    }
  }

  return null
}

async function applyLoginShellPath({
  env = process.env,
  platform = process.platform,
  execFileFn = execFile,
  timeoutMs = ATTEMPT_TIMEOUT_MS
}: any = {}) {
  if (platform === 'win32') {
    return { applied: false, reason: 'win32' }
  }

  const loginPath = await captureLoginShellPath({ env, platform, execFileFn, timeoutMs })

  if (!loginPath) {
    return { applied: false, reason: 'unresolved' }
  }

  const key = pathEnvKey(env, platform)
  const delimiter = delimiterForPlatform(platform)
  const merged = mergeLoginShellPath(loginPath, env?.[key] || '', { delimiter })

  if (!merged || merged === env?.[key]) {
    return { applied: false, reason: 'unchanged', path: merged }
  }

  env[key] = merged

  return { applied: true, path: merged }
}

// Single-flight: the warmup at app start and the await before the backend
// spawn share one resolution. Never rejects.
let _ensurePromise: Promise<any> | null = null

function ensureLoginShellPath(options: any = {}) {
  if (!_ensurePromise) {
    _ensurePromise = applyLoginShellPath(options).catch(error => ({
      applied: false,
      reason: String(error?.message || error)
    }))
  }

  return _ensurePromise
}

function resetLoginShellPathForTests() {
  _ensurePromise = null
}

export {
  applyLoginShellPath,
  captureLoginShellPath,
  ensureLoginShellPath,
  extractSentinelPath,
  loginShellExecutable,
  mergeLoginShellPath,
  resetLoginShellPathForTests
}
