import { describe, expect, it, vi } from 'vitest'

import {
  isUnderInstallRoot,
  listWindowsProcesses,
  reapPackageRootedProcesses,
  type RunningProcess
} from './package-process-reap'

const ROOT = 'C:\\Program Files\\WindowsApps\\NousResearch.HermesBundled_0.21.20.25635_arm64__e60prshbsznhj'

const GPG_AGENT: string = `${ROOT}\\app\\tools\\git\\gpg-agent.exe`
const PAYLOAD_PYTHON: string = `${ROOT}\\app\\tools\\python\\python.exe`
const MAIN_EXE: string = `${ROOT}\\app\\Hermes.exe`
const TOOLS_ROOT: string = 'C:\\Hermes\\tools'
const STORE_NODE: string = `${TOOLS_ROOT}\\node\\node.exe`
const DAEMON_CMDLINE: string = 'gpg-agent --daemon'

function proc(
  pid: number,
  path: string | null,
  commandLine: string | null = DAEMON_CMDLINE,
  parentPid: number | null = null
): RunningProcess {
  return { pid, parentPid, path, commandLine }
}

describe('isUnderInstallRoot', () => {
  it('matches an image nested under the root regardless of case or separator', () => {
    expect(isUnderInstallRoot(GPG_AGENT, ROOT)).toBe(true)
    expect(isUnderInstallRoot(GPG_AGENT.toUpperCase(), ROOT)).toBe(true)
    expect(isUnderInstallRoot(GPG_AGENT.replace(/\\/g, '/'), ROOT)).toBe(true)
    expect(isUnderInstallRoot(MAIN_EXE, `${ROOT}\\`)).toBe(true)
  })

  it('does NOT match a sibling package whose name merely shares the prefix', () => {
    // A bare startsWith would kill another package's processes here. The
    // separator check is the whole guard.
    const sibling =
      'C:\\Program Files\\WindowsApps\\NousResearch.HermesBundled_0.21.20.256350_arm64__e60prshbsznhj\\app\\Hermes.exe'

    expect(isUnderInstallRoot(sibling, ROOT)).toBe(false)
  })

  it('does not match another vendor, an unreadable path, or an empty root', () => {
    const other =
      'C:\\Program Files\\WindowsApps\\8bitSolutionsLLC.bitwardendesktop_2026.7.0.0_arm64__x\\app\\Bitwarden.exe'

    expect(isUnderInstallRoot(other, ROOT)).toBe(false)
    expect(isUnderInstallRoot(null, ROOT)).toBe(false)
    expect(isUnderInstallRoot(MAIN_EXE, null)).toBe(false)
    expect(isUnderInstallRoot(MAIN_EXE, '')).toBe(false)
    expect(isUnderInstallRoot(MAIN_EXE, [])).toBe(false)
  })

  it('matches under ANY supplied root, so both install shapes are covered', () => {
    // Bundled artifact resources AND a mutable install's managed tool store.
    const roots = [ROOT, TOOLS_ROOT]

    expect(isUnderInstallRoot(GPG_AGENT, roots)).toBe(true)
    expect(isUnderInstallRoot(STORE_NODE, roots)).toBe(true)
    expect(isUnderInstallRoot('C:\\Windows\\System32\\node.exe', roots)).toBe(false)
  })

  it('skips nullish roots instead of matching everything', () => {
    // A bootstrap install has no resourcesPath payload: the artifact root is
    // absent and only the tool store is real. An absent root must never widen
    // the predicate.
    expect(isUnderInstallRoot(STORE_NODE, [null, TOOLS_ROOT])).toBe(true)
    expect(isUnderInstallRoot('C:\\Windows\\System32\\node.exe', [null, undefined, ''])).toBe(false)
  })
})

interface SelectionCase {
  name: string
  processes: RunningProcess[]
  roots?: Array<string | null>
  selfPid?: number
  excludePids?: number[]
  killed: number[]
}

const cases: SelectionCase[] = [
  { name: 'owned daemon', processes: [proc(10, GPG_AGENT)], killed: [10] },
  { name: 'self daemon', processes: [proc(10, GPG_AGENT)], selfPid: 10, killed: [] },
  { name: 'excluded daemon', processes: [proc(10, GPG_AGENT), proc(11, GPG_AGENT)], excludePids: [10], killed: [11] },
  { name: 'sibling-prefix daemon', processes: [proc(10, `${ROOT}-other\\gpg-agent.exe`)], killed: [] },
  { name: 'external daemon', processes: [proc(10, 'C:\\Git\\gpg-agent.exe'), proc(11, GPG_AGENT)], killed: [11] },
  {
    name: 'unidentified parent and daemon descendants',
    processes: [proc(12, GPG_AGENT, DAEMON_CMDLINE, 11), proc(11, GPG_AGENT, DAEMON_CMDLINE, 10), proc(10, null)],
    killed: []
  },
  {
    name: 'shared runtime and daemon descendants',
    processes: [
      proc(12, GPG_AGENT, DAEMON_CMDLINE, 11),
      proc(11, GPG_AGENT, DAEMON_CMDLINE, 10),
      proc(10, PAYLOAD_PYTHON, '-u -X utf8 -m gateway.run')
    ],
    killed: []
  },
  {
    name: 'shared entrypoints irrespective of argv',
    processes: [
      proc(10, MAIN_EXE, '--app'),
      proc(11, PAYLOAD_PYTHON, null),
      proc(12, STORE_NODE, 'daemon.js'),
      proc(13, GPG_AGENT)
    ],
    roots: [ROOT, TOOLS_ROOT],
    killed: [13]
  },
  {
    name: 'mutable tool roots',
    processes: [proc(10, `${TOOLS_ROOT}\\git\\gpg-agent.exe`), proc(11, GPG_AGENT)],
    roots: [null, TOOLS_ROOT],
    killed: [10]
  },
  { name: 'unreadable path', processes: [proc(10, null)], killed: [] }
]

it.each(cases)(
  'safe kill selection: $name',
  ({ processes, roots = [ROOT], selfPid = 1, excludePids = [], killed }: SelectionCase): void => {
    const signalled: number[] = []

    const outcome: ReturnType<typeof reapPackageRootedProcesses> = reapPackageRootedProcesses({
      installRoots: roots,
      selfPid,
      excludePids,
      isWindows: true,
      listProcesses: (): RunningProcess[] => processes,
      killProcess: (pid: number): void => {
        signalled.push(pid)
      }
    })

    expect(outcome.killed).toEqual(killed)
    expect(outcome.matched).toBe(killed.length)
    expect(signalled).toEqual(killed)
  }
)

it('continues after kill failure and refuses absent roots or failed enumeration', (): void => {
  const processes: RunningProcess[] = [proc(50, GPG_AGENT), proc(51, GPG_AGENT)]

  const base: Parameters<typeof reapPackageRootedProcesses>[0] = {
    installRoots: [ROOT],
    isWindows: true,
    selfPid: 1,
    listProcesses: (): RunningProcess[] => processes,
    killProcess: (pid: number): void => {
      if (pid === 50) {
        throw new Error('Access denied')
      }
    }
  }

  expect(reapPackageRootedProcesses(base)).toMatchObject({ failed: [50], killed: [51] })

  for (const overrides of [
    { isWindows: false },
    { installRoots: [null, ''] },
    {
      listProcesses: (): never => {
        throw new Error('enumeration failed')
      }
    }
  ]) {
    const killProcess = vi.fn<(pid: number) => void>()
    expect(reapPackageRootedProcesses({ ...base, ...overrides, killProcess })).toMatchObject({
      skipped: true,
      killed: []
    })
    expect(killProcess).not.toHaveBeenCalled()
  }
})

describe('listWindowsProcesses', () => {
  it('parses Win32_Process output (pid|parent|path|command line), mapping unreadable fields to null', () => {
    // The real shape: ExecutablePath / CommandLine are null for processes we
    // cannot open, so the script emits empty middle/tail fields.
    const stdout = [`48236|61728|${GPG_AGENT}|gpg-agent --daemon`, `22660|4|${MAIN_EXE}|`, '4|0||', ''].join('\r\n')

    const parsed = listWindowsProcesses(() => stdout)

    expect(parsed).toEqual([
      { pid: 48236, parentPid: 61728, path: GPG_AGENT, commandLine: 'gpg-agent --daemon' },
      { pid: 22660, parentPid: 4, path: MAIN_EXE, commandLine: '' },
      { pid: 4, parentPid: 0, path: null, commandLine: '' }
    ])
  })

  it('keeps paths and command lines that contain pipes and spaces intact', () => {
    // Splitting on the FIRST three separators is what preserves the rest —
    // a command line may contain "|", a path never does.
    const cmd = 'python -c "x = a | b"'
    const parsed = listWindowsProcesses(() => `10|99|${MAIN_EXE}|${cmd}`)

    expect(parsed[0].path).toBe(MAIN_EXE)
    expect(parsed[0].commandLine).toBe(cmd)
    expect(parsed[0].parentPid).toBe(99)
  })

  it('ignores malformed lines instead of inventing pids', () => {
    const parsed = listWindowsProcesses(() => ['garbage', '|no-pid', 'abc|path', ''].join('\n'))

    expect(parsed).toEqual([])
  })

  it('runs hidden and bounded so it cannot stall or flash a console on quit', () => {
    const calls: Array<{ file: string; options: { timeout: number; windowsHide: boolean } }> = []

    const execFile = (file: string, _args: string[], options: { timeout: number; windowsHide: boolean }): string => {
      calls.push({ file, options })

      return ''
    }

    listWindowsProcesses(execFile)

    expect(calls[0].file).toBe('powershell.exe')
    expect(calls[0].options.windowsHide).toBe(true)
    expect(calls[0].options.timeout).toBeGreaterThan(0)
  })
})
