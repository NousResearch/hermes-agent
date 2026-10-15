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
