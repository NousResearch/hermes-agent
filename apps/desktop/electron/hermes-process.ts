import { execSync } from 'node:child_process'
import fs from 'node:fs'

/**
 * The Hermes runtime executable, matched as a whole argv TOKEN's basename — never an
 * argv substring (root AGENTS.md HX003 forbids argv-substring process identity). A
 * substring check over the full command line matched any process whose path merely
 * contained "hermes" (`vim ~/hermes-notes.txt`, a `/opt/hermesbrew/...` tool), so a
 * recycled PID could be mistaken for a live gateway. Mirrors the whole-token basename
 * policy in `isSharedRuntime` (package-process-reap.ts) and the Python canonical
 * matchers named in AGENTS.md HX003.
 */
const HERMES_RUNTIME_BASENAME = /^hermes(?:-agent|-acp)?(?:\.exe|\.cmd)?$/i

/** Tokenize a raw command line: `/proc/<pid>/cmdline` is NUL-separated, `ps -o args=`
 *  is whitespace-separated. Either way, split on both so one matcher serves both. */
function commandLineTokens(raw: string): string[] {
  return raw
    .split(/[\0\s]+/)
    .filter(token => token.length > 0)
}

/** True when `raw` is a Hermes runtime command line: some whole argv token's basename
 *  is the Hermes executable (`hermes`, launched directly or by an interpreter such as
 *  `python …/hermes gateway run`). Exported for unit tests. */
export function commandLineIsHermesRuntime(raw: string): boolean {
  return commandLineTokens(raw).some(token => HERMES_RUNTIME_BASENAME.test(token.replace(/^.*[\\/]/, '')))
}

// True when the given pid belongs to a running Hermes process, identified by a whole
// argv token (not a substring), avoiding false positives from stale gateway.pid files
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

    return commandLineIsHermesRuntime(cmdline)
  } catch {
    // /proc not available (macOS) — fall back to ps. Use -o args= to inspect
    // the full command line, not just the process name.  -o comm= would return
    // "python3" for any Python process, creating false positives.
    try {
      const out = execSync(`ps -p ${pid} -o args=`, { encoding: 'utf8', timeout: 2000 })

      return commandLineIsHermesRuntime(out)
    } catch {
      return false
    }
  }
}
