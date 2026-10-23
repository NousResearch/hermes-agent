/**
 * process-identity.ts
 *
 * "Is this PID a live hermes process?" — the identity half of the stale
 * `gateway.pid` guard. Liveness alone is not identity: the OS recycles PIDs,
 * so a stale pid file can name a live but unrelated process, and
 * `findRunningGatewayProfiles` would report a gateway that is not ours.
 *
 * Extracted from main.ts so the per-platform probes are testable without
 * booting Electron (the same reason backend-claim.ts was extracted).
 *
 * Each platform needs its own probe because no two share one:
 *   linux  — /proc/<pid>/cmdline
 *   darwin — BSD `ps -p <pid> -o args=`
 *   win32  — Get-CimInstance Win32_Process.CommandLine
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'

import { hiddenWindowsChildOptions } from './windows-child-options'

/** PowerShell 5.1 cold starts routinely exceed a 2s budget (#87169). */
const WINDOWS_PROBE_TIMEOUT_MS = 5_000

const POSIX_PROBE_TIMEOUT_MS = 2_000

export function isHermesProcess(pid: number): boolean {
  try {
    process.kill(pid, 0) // signal 0 = existence check, no signal sent
  } catch {
    return false
  }

  if (process.platform === 'win32') {
    return windowsCommandLine(pid)?.toLowerCase().includes('hermes') ?? false
  }

  // On macOS / Linux, check the command line to avoid PID recycling false positives.
  try {
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8')

    return cmdline.includes('hermes')
  } catch {
    // /proc not available (macOS) — fall back to ps. Use -o args= to inspect
    // the full command line, not just the process name.  -o comm= would return
    // "python3" for any Python process, creating false positives.
    try {
      const out = execFileSync('ps', ['-p', String(pid), '-o', 'args='], {
        ...hiddenWindowsChildOptions(),
        encoding: 'utf8',
        timeout: POSIX_PROBE_TIMEOUT_MS
      })

      return out.includes('hermes')
    } catch {
      return false
    }
  }
}

/**
 * `Win32_Process.CommandLine` for `pid`, or null when the probe fails.
 *
 * Windows is the one platform with neither `/proc` nor a usable `ps`: the
 * procps the app ships rejects the BSD-style `-o` that macOS accepts
 * (`ps: unknown option -- o`), so the macOS fallback above could never
 * identify a process there. Every call threw, the catch returned `false`, and
 * `findRunningGatewayProfiles` reported no running gateway on Windows at all.
 *
 * Matched case-insensitively because Windows paths are: `Hermes.exe` is the
 * same executable as `hermes.exe`.
 */
function windowsCommandLine(pid: number): string | null {
  try {
    return execFileSync(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}').CommandLine`
      ],
      { ...hiddenWindowsChildOptions(), encoding: 'utf8', timeout: WINDOWS_PROBE_TIMEOUT_MS }
    )
  } catch {
    return null
  }
}
