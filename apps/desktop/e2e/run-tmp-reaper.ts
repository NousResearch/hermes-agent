/**
 * Stop the backends E2E runs start, at three points.
 *
 * Desktop dials its backend through `hermes gateway ensure`, which spawns a detached
 * `gateway run` that outlives the app on purpose (production attach mode). Closing the app
 * therefore never stops it, and an orphan rewrites its sandbox HERMES_HOME after the
 * fixture's rmSync, so the leaked run root reappears. Each layer below covers a teardown the
 * one before it cannot reach:
 *
 *  - per sandbox, `stopSandboxGatewayDaemons`: the daemons a sandbox's `gateway.lock` files
 *    record (every platform);
 *  - per run, at runner exit: every process whose environment names this run's root (a spec
 *    whose fixture never returned, profile gateways outside the lock homes);
 *  - per host, at the next runner's start: every process left under a run root whose runner
 *    is dead (SIGKILLed by a CI or agent timeout, so neither teardown nor the exit hook ran).
 *
 * The run-level layers read /proc (Linux only; elsewhere they are no-ops). They only signal a
 * process whose environment carries BOTH `HERMES_E2E_RUN_TMP=<root>` and a HERMES_HOME under
 * that root, so a developer's own gateway (no run root) is never a candidate, and a root whose
 * runner is alive (a concurrent run) or unrecorded (an older harness) is left alone.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { isSandboxProcess, procfsCensus, readProc } from './core/process-census'

const OWNER_FILE = '.runner.json'

function sleepSync(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)
}

/**
 * SIGTERM `pids`, then SIGKILL whatever `stillOurs` still claims after `graceMs`. Re-checking
 * `stillOurs` before every signal keeps a pid recycled mid-wait from ever being touched.
 */
export function terminateProcesses(pids: Iterable<number>, stillOurs: (pid: number) => boolean, graceMs = 15_000): void {
  const live = new Set(pids)

  const signal = (sig: NodeJS.Signals) => {
    for (const pid of live) {
      try {
        if (stillOurs(pid)) {
          process.kill(pid, sig)
        } else {
          live.delete(pid)
        }
      } catch {
        live.delete(pid)
      }
    }
  }

  signal('SIGTERM')

  for (const deadline = Date.now() + graceMs; live.size && Date.now() < deadline; ) {
    sleepSync(100)

    for (const pid of [...live]) {
      if (!stillOurs(pid)) {
        live.delete(pid)
      }
    }
  }

  signal('SIGKILL')
}

function processHomeUnder(pid: number, root: string, hermesHome: string): boolean {
  if (!procfsCensus) {
    return isSandboxProcess(pid, hermesHome)
  }

  const info = readProc(pid)

  return Boolean(info?.cmdline) && info!.environ.split('\0').some(entry => entry.startsWith(`HERMES_HOME=${root}`))
}

/**
 * SIGTERM (then SIGKILL after 15 s) every daemon recorded in a sandbox `gateway.lock` (launch
 * home and each profile home), while its environment still names this sandbox.
 */
export function stopSandboxGatewayDaemons(root: string, hermesHome: string): void {
  const profilesDir = path.join(hermesHome, 'profiles')

  const homes = [
    hermesHome,
    ...(fs.existsSync(profilesDir) ? fs.readdirSync(profilesDir).map(name => path.join(profilesDir, name)) : [])
  ]

  const pids = new Set<number>()

  for (const home of homes) {
    try {
      const pid = Number(JSON.parse(fs.readFileSync(path.join(home, 'gateway.lock'), 'utf8'))?.pid)

      if (Number.isInteger(pid) && pid > 0 && processHomeUnder(pid, root, hermesHome)) {
        pids.add(pid)
      }
    } catch {
      // No lock (remote-only spec, never booted) or an empty one: nothing to stop.
    }
  }

  terminateProcesses(pids, pid => processHomeUnder(pid, root, hermesHome))
}

/** Kernel start time of `pid` (/proc/<pid>/stat field 22): with the pid, a process identity no recycle repeats. */
function startTicks(pid: number): null | string {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')

    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? null
  } catch {
    return null
  }
}

/** Record the calling process as the runner that owns `root`. */
export function recordRunOwner(root: string): void {
  fs.writeFileSync(path.join(root, OWNER_FILE), JSON.stringify({ pid: process.pid, start: startTicks(process.pid) }))
}

/** False only when `root`'s recorded runner is provably gone (a root with no record counts as live). */
function runnerAlive(root: string): boolean {
  let owner: { pid?: unknown; start?: unknown }

  try {
    owner = JSON.parse(fs.readFileSync(path.join(root, OWNER_FILE), 'utf8'))
  } catch {
    return true
  }

  return typeof owner.pid === 'number' && owner.start !== null && startTicks(owner.pid) === owner.start
}

/** The run root a process belongs to: its HERMES_E2E_RUN_TMP, when its HERMES_HOME also lies under it. */
function runRootOf(pid: number): null | string {
  const env = readProc(pid)?.environ.split('\0') ?? []
  const root = env.find(entry => entry.startsWith('HERMES_E2E_RUN_TMP='))?.slice('HERMES_E2E_RUN_TMP='.length)
  const home = env.find(entry => entry.startsWith('HERMES_HOME='))?.slice('HERMES_HOME='.length)

  return root && home?.startsWith(root + path.sep) ? root : null
}

/**
 * Stop every process left under a run root `shouldReap` selects (Linux /proc only), then remove
 * those roots. Returns the roots reaped.
 */
export function reapRunBackends(shouldReap: (root: string) => boolean): string[] {
  if (!procfsCensus) {
    return []
  }

  const verdicts = new Map<string, boolean>()
  const victims = new Map<number, string>()

  for (const entry of fs.readdirSync('/proc')) {
    const pid = Number(entry)
    const root = Number.isInteger(pid) && pid !== process.pid ? runRootOf(pid) : null

    if (root) {
      if (!verdicts.has(root)) {
        verdicts.set(root, shouldReap(root))
      }

      if (verdicts.get(root)) {
        victims.set(pid, root)
      }
    }
  }

  terminateProcesses(victims.keys(), pid => runRootOf(pid) === victims.get(pid))
  const roots = [...new Set(victims.values())]

  for (const root of roots) {
    fs.rmSync(root, { recursive: true, force: true })
  }

  return roots
}

/**
 * Reap the backends of every run whose runner is dead: host-wide, or only the runs whose root
 * lies under `scope` (a test reaps its own fixtures, never another abandoned run on the host).
 */
export function reapAbandonedRuns(scope?: string): string[] {
  return reapRunBackends(root => (scope === undefined || root.startsWith(scope + path.sep)) && !runnerAlive(root))
}
