import { execSync } from 'node:child_process'
import fs from 'node:fs'

// True when the given pid belongs to a running process whose command line
// contains "hermes", avoiding false positives from stale gateway.pid files
// whose PID was recycled by the OS to an unrelated process.
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
      const out = execSync(`ps -p ${pid} -o args=`, { encoding: 'utf8', timeout: 2000 })

      return out.includes('hermes')
    } catch {
      return false
    }
  }
}
