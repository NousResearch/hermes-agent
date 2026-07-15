import { execFileSync } from 'node:child_process'
import path from 'node:path'

const SHORT_PATH_ENV = 'HERMES_GIT_BINARY_PATH'
const SHORT_PATH_COMMAND = `for %A in ("%${SHORT_PATH_ENV}%") do @echo %~sA`

const windowsRoot = [process.env.SystemRoot, process.env.WINDIR].find((candidate): candidate is string =>
  Boolean(candidate && path.win32.isAbsolute(candidate))
)

const CMD_EXE =
  process.env.ComSpec && path.win32.isAbsolute(process.env.ComSpec)
    ? process.env.ComSpec
    : path.win32.join(windowsRoot || String.raw`C:\Windows`, 'System32', 'cmd.exe')

interface ShortPathExecOptions {
  encoding: 'utf8'
  env: NodeJS.ProcessEnv
  timeout: number
  windowsHide: boolean
  windowsVerbatimArguments: boolean
}

type ShortPathExec = (command: string, args: string[], options: ShortPathExecOptions) => Buffer | string

const defaultShortPathExec = execFileSync as ShortPathExec
const shortPathCache = new Map<string, string>()

// Matches simple-git's custom-binary allowlist. Keep this local guard so the
// Windows short-path result is accepted only when simple-git will accept it.
function isSimpleGitSafeBinary(binary: string): boolean {
  return /^([a-z]:)?([a-z0-9/.\\_~-]+)$/i.test(binary)
}

/**
 * Convert a spaced Windows executable path to its 8.3 form for simple-git.
 *
 * `windowsVerbatimArguments` is required here. Without it Node re-quotes the
 * command passed to `cmd.exe /c`, and `%~sA` echoes a malformed long path rather
 * than performing short-name expansion. The executable path travels through an
 * environment variable so it is never interpolated into cmd syntax.
 *
 * Volumes may disable 8.3 names. In that case the original path is returned and
 * the caller can use simple-git's trusted-custom-binary escape hatch.
 */
export function toWindowsShortPath(filePath: string, exec: ShortPathExec = defaultShortPathExec): string {
  if (!/\s/.test(filePath)) {
    return filePath
  }

  // The main process resolves one git binary for its lifetime. Cache only the
  // production executor path; injected test executors must remain isolated.
  if (exec === defaultShortPathExec) {
    const cached = shortPathCache.get(filePath)

    if (cached) {
      return cached
    }
  }

  let resolved = filePath

  try {
    const result = String(
      exec(CMD_EXE, ['/d', '/s', '/c', SHORT_PATH_COMMAND], {
        encoding: 'utf8',
        env: { ...process.env, [SHORT_PATH_ENV]: filePath },
        timeout: 5_000,
        windowsHide: true,
        windowsVerbatimArguments: true
      })
    ).trim()

    if (path.win32.isAbsolute(result) && isSimpleGitSafeBinary(result)) {
      resolved = result
    }
  } catch {
    // 8.3 names may be unavailable; the caller has a safe fallback.
  }

  if (exec === defaultShortPathExec) {
    shortPathCache.set(filePath, resolved)
  }

  return resolved
}
