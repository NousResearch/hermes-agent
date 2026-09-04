/**
 * Cross-platform "full command line for a PID" lookup (#102660).
 *
 * Never a bare ``ps`` on Windows: there is no ps there, and a bare PATH
 * lookup resolves to unrelated executables — System32 hosts PS.exe, the
 * legacy planned-shutdown tool, which is how a Hermes desktop build ended up
 * executing PS.exe from this exact site. PowerShell's Win32_Process
 * CommandLine is the Windows equivalent of the POSIX ps full-command-line
 * probe (the same probe backendCommandForPid uses).
 *
 * Both branches execute a fixed binary with a literal argv list (the pid
 * rides as its own argv element, never inside a shell command string), so no
 * platform routes this through a shell wrapper and a corrupted pid value can
 * never alter which program runs. Platform policy is dependency-free and
 * unit-testable the same way backend-claim.ts's probe policy is (#93608).
 */
import { execFileSync } from 'node:child_process'

/**
 * Synchronous full-command-line read for a live pid; null on any failure
 * (missing process, probe timeout, invalid pid).
 */
export function readProcessCommandLineSync(pid: number, platform: string = process.platform): string | null {
  if (!Number.isInteger(pid) || pid <= 0) {
    return null
  }
  try {
    if (platform === 'win32') {
      const out = execFileSync('powershell.exe', [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        '(Get-CimInstance Win32_Process -Filter ("ProcessId = " + $args[0])).CommandLine',
        String(pid)
      ], {
        encoding: 'utf8',
        // PowerShell 5.1 cold starts routinely exceed 2s (#87169); a timeout
        // degrades to null ("unknown") rather than hanging the caller.
        timeout: 5_000
      })
      return typeof out === 'string' ? out : null
    }
    const out = execFileSync('ps', ['-p', String(pid), '-o', 'args='], { encoding: 'utf8', timeout: 2_000 })
    return typeof out === 'string' ? out : null
  } catch {
    return null
  }
}
