/**
 * Process census for the core lane, per platform.
 *
 * Linux reads /proc: a process belongs to a sandbox when its environment names the sandbox
 * HERMES_HOME (orphans reparented to init included). Other platforms have no readable foreign
 * environment, so membership is the process tree rooted at every gateway pid a sandbox
 * `gateway.lock` has ever recorded (remembered per sandbox, so the census still sees a stopped
 * gateway's orphans after its lock is gone). Windows keeps a dead parent's id in
 * ParentProcessId, so the tree walk still reaches orphans there.
 *
 * `HERMES_E2E_PROC_CENSUS=ps` forces the tree census on Linux (how the non-/proc path is tested).
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

export interface ProcInfo {
  pid: number
  ppid: number
  cmdline: string
}

export const procfsCensus = process.env.HERMES_E2E_PROC_CENSUS !== 'ps' && fs.existsSync('/proc/self/environ')

export function readProc(pid: number): null | { environ: string; cmdline: string; ppid: number } {
  try {
    const environ = fs.readFileSync(`/proc/${pid}/environ`, 'utf8')
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ').trim()
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
    // Field 4 (ppid) follows the parenthesised comm, which may contain spaces.
    const ppid = Number(stat.slice(stat.lastIndexOf(')') + 2).split(' ')[1])

    return { environ, cmdline, ppid }
  } catch {
    return null
  }
}

/** Every live process (pid, ppid, command line); zombies/unreadable entries skipped. */
export function listProcesses(): ProcInfo[] {
  if (procfsCensus) {
    return fs.readdirSync('/proc').flatMap(entry => {
      const pid = Number(entry)
      const info = Number.isInteger(pid) ? readProc(pid) : null

      return info?.cmdline ? [{ pid, ppid: info.ppid, cmdline: info.cmdline }] : []
    })
  }

  const listing =
    process.platform === 'win32'
      ? spawnSync(
          'powershell.exe',
          [
            '-NoProfile',
            '-NonInteractive',
            '-Command',
            'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)" }'
          ],
          { encoding: 'utf8', timeout: 60_000, windowsHide: true }
        ).stdout
      : spawnSync('ps', ['-axww', '-o', 'pid=,ppid=,args='], { encoding: 'utf8', timeout: 30_000 }).stdout

  return String(listing ?? '')
    .split(/\r?\n/)
    .flatMap(line => {
      const match = process.platform === 'win32' ? /^(\d+)\t(\d+)\t(.*)$/.exec(line) : /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line)

      return match && match[3].trim() ? [{ pid: Number(match[1]), ppid: Number(match[2]), cmdline: match[3].trim() }] : []
    })
}

const recordedRoots = new Map<string, Set<number>>()

/** Gateway pids recorded in the sandbox's `gateway.lock` files (launch home + each profile home). */
function lockedGatewayPids(hermesHome: string): number[] {
  const profilesDir = path.join(hermesHome, 'profiles')
  const homes = [hermesHome, ...(fs.existsSync(profilesDir) ? fs.readdirSync(profilesDir).map(name => path.join(profilesDir, name)) : [])]

  return homes.flatMap(home => {
    try {
      const pid = Number(JSON.parse(fs.readFileSync(path.join(home, 'gateway.lock'), 'utf8'))?.pid)

      return Number.isInteger(pid) && pid > 0 ? [pid] : []
    } catch {
      return []
    }
  })
}

/** Every live process belonging to the sandbox whose HERMES_HOME is `hermesHome` (orphans included). */
export function sandboxProcessesOf(hermesHome: string, exclude = process.pid): ProcInfo[] {
  if (procfsCensus) {
    const needle = `HERMES_HOME=${hermesHome}\0`

    return fs.readdirSync('/proc').flatMap(entry => {
      const pid = Number(entry)
      const info = Number.isInteger(pid) && pid !== exclude ? readProc(pid) : null

      // Zombies have an empty cmdline and are already dead for our purposes.
      return info?.cmdline && (info.environ + '\0').includes(needle) ? [{ pid, ppid: info.ppid, cmdline: info.cmdline }] : []
    })
  }

  const roots = recordedRoots.get(hermesHome) ?? new Set<number>()
  recordedRoots.set(hermesHome, roots)
  const all = listProcesses()
  const live = new Set(all.map(proc => proc.pid))

  for (const pid of lockedGatewayPids(hermesHome)) {
    if (live.has(pid)) {
      roots.add(pid)
    }
  }

  const members = new Set([...roots].filter(pid => live.has(pid)))
  const parents = new Set(roots)

  for (let grew = true; grew; ) {
    grew = false

    for (const proc of all) {
      if (!parents.has(proc.pid) && parents.has(proc.ppid)) {
        parents.add(proc.pid)
        members.add(proc.pid)
        grew = true
      }
    }
  }

  return all.filter(proc => members.has(proc.pid) && proc.pid !== exclude)
}

/** True while `pid` is still a live member of the sandbox (a recycled pid never matches). */
export function isSandboxProcess(pid: number, hermesHome: string): boolean {
  return sandboxProcessesOf(hermesHome).some(proc => proc.pid === pid)
}

/** Every live process whose command line carries `tag`. */
export function processesTagged(tag: string): ProcInfo[] {
  return listProcesses().filter(proc => proc.pid !== process.pid && proc.cmdline.includes(tag))
}

/** `value` single-quoted for the POSIX shell the terminal tool runs (Git Bash on Windows). */
export function shellQuote(value: string): string {
  const posix = process.platform === 'win32' ? value.replace(/^([A-Za-z]):[\\/]/, (_, d: string) => `/${d.toLowerCase()}/`).replace(/\\/g, '/') : value

  return `'${posix.replace(/'/g, `'\\''`)}'`
}
