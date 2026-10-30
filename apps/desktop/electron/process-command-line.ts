/**
 * process-command-line.ts
 *
 * The command line of a running process, read the way the platform can
 * actually answer it.
 *
 * Why this exists: `isHermesProcess` in main.ts guarded its /proc read with
 * "if /proc is unavailable, fall back to ps" -- but on Windows NEITHER is
 * available. There is no /proc, and `ps` is not a Windows program, so the
 * fallback always failed. It failed quietly (the catch returned false), but
 * the shell's "not recognized" error leaked into the console the user sees,
 * and the function answered "not a Hermes process" for every live gateway.
 *
 * The Windows probe is the same one main.ts already uses for the backend
 * claim (`backendCommandForPid`): Get-CimInstance Win32_Process, filtered by
 * ProcessId, reading .CommandLine. So the behaviour is already proven in
 * this codebase -- it was simply not applied here.
 *
 * The probe is SYNCHRONOUS because its only caller is synchronous, and it
 * runs once per desktop launch. The child process is spawned with an
 * explicit `stdio`, because execFileSync's default sends the child's stderr
 * straight to the parent's stderr -- that is exactly how the shell's "not
 * recognized" message reached the user's console. Capturing stderr here
 * keeps a failed probe silent.
 *
 * Kept in its own dependency-free module (no electron import) so it can be
 * unit-tested directly on every platform, the way windows-child-options.ts
 * and parent-process-identity.ts are.
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'

/** How long the OS probe may take before it is treated as "unknown". */
const COMMAND_LINE_TIMEOUT_MS = 2000

export interface ProcessCommandLineDeps {
  /** The platform: 'win32' selects the PowerShell probe, anything else /proc. */
  platform?: NodeJS.Platform
  /** Read a file (used for /proc/<pid>/cmdline). */
  readFileSync?: (path: string, encoding: 'utf8') => string
  /** Run a command and return its stdout (used for the ps / PowerShell probe). */
  execFileImpl?: typeof execFileSync
  /** Is this pid alive? */
  isAlive?: (pid: number) => boolean
}

/** True when the pid belongs to a running process. */
function defaultIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0) // signal 0 = existence check, no signal sent

    return true
  } catch {
    return false
  }
}

/**
 * The command line of a running process, or null when it cannot be read.
 *
 * Returns null (never throws) for a dead pid, a probe timeout, or a platform
 * that cannot answer the question. A null answer means "unknown", and every
 * caller must treat it as "not confirmed" rather than "confirmed negative".
 */
export function processCommandLine(pid: number, deps: ProcessCommandLineDeps = {}): string | null {
  const platform = deps.platform ?? process.platform
  const readFileSync = deps.readFileSync ?? ((path: string, encoding: 'utf8') => fs.readFileSync(path, encoding))
  const execFileImpl = deps.execFileImpl ?? execFileSync
  const isAlive = deps.isAlive ?? defaultIsAlive

  if (!Number.isInteger(pid) || pid <= 0 || !isAlive(pid)) {
    return null
  }

  if (platform === 'win32') {
    return windowsCommandLine(pid, execFileImpl)
  }

  // Linux: /proc/<pid>/cmdline is authoritative and needs no subprocess.
  try {
    const raw = readFileSync(`/proc/${pid}/cmdline`, 'utf8')

    return raw.replace(/\0/g, ' ')
  } catch {
    // No /proc (macOS) -- ask the OS instead.
    return posixCommandLine(pid, execFileImpl)
  }
}

/**
 * Windows: Get-CimInstance reads the command line of any process the caller
 * can see. stderr is captured, never printed, so a probe cannot spam the
 * user's console.
 */
function windowsCommandLine(pid: number, execFileImpl: typeof execFileSync): string | null {
  try {
    const out = execFileImpl(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}').CommandLine`,
      ],
      {
        encoding: 'utf8',
        timeout: COMMAND_LINE_TIMEOUT_MS,
        // Capture stderr instead of letting it reach the parent's console.
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      },
    )

    const text = String(out || '').trim()

    return text || null
  } catch {
    // The process is gone, unreadable, or PowerShell failed: unknown.
    return null
  }
}

/**
 * macOS / other POSIX: `ps` prints the full command line with -o args=.
 * -o comm= would return "python3" for any Python process, creating false
 * positives.
 */
function posixCommandLine(pid: number, execFileImpl: typeof execFileSync): string | null {
  try {
    const out = execFileImpl('ps', ['-p', String(pid), '-o', 'args='], {
      encoding: 'utf8',
      timeout: COMMAND_LINE_TIMEOUT_MS,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const text = String(out || '').trim()

    return text || null
  } catch {
    return null
  }
}
