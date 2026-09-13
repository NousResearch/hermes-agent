import { describe, expect, it, vi } from 'vitest'

import {
  isUnderInstallRoot,
  listWindowsProcesses,
  protectedRuntimePids,
  reapPackageRootedProcesses,
  type RunningProcess
} from './package-process-reap'

const ROOT = 'C:\\Program Files\\WindowsApps\\NousResearch.HermesBundled_0.21.20.25635_arm64__e60prshbsznhj'

/** A mutable install's managed tool store: HERMES_RUNTIME_DIR / <root>\tools. */
const TOOLS_ROOT = 'C:\\Users\\arilo\\AppData\\Local\\hermes\\tools'

/** The live-observed pinner: payload git's gpg-agent, daemonized out of our tree. */
const GPG_AGENT = `${ROOT}\\app\\resources\\agent-payload\\tools\\git-2.53.0+3-win32-arm64\\usr\\bin\\gpg-agent.exe`
const PAYLOAD_PYTHON = `${ROOT}\\app\\resources\\agent-payload\\tools\\python-3.11.16+20260814-win32-arm64\\python.exe`
const MAIN_EXE = `${ROOT}\\app\\Hermes.exe`

/** Live-observed daemon command line (payload git gpg-agent, C09). */
const DAEMON_CMDLINE = `"${GPG_AGENT}" --use-standard-socket --daemon`

/** Same class, mutable install: pm-staged node holding its own image open. */
const STORE_NODE = `${TOOLS_ROOT}\\node-26.7.0-win32-arm64\\node.exe`

/**
 * The gateway that survives desktop quit: payload python running the Hermes
 * gateway under the user-logon Scheduled Task (bundled installs run it out
 * of the payload, so its image IS under the artifact root).
 */
const GATEWAY_CMDLINE = `"${PAYLOAD_PYTHON}" -m hermes_cli.main gateway run`
/** Legacy/alternate module launcher of the same runtime. */
const GATEWAY_RUN_MODULE_CMDLINE = `"${PAYLOAD_PYTHON}" -m gateway.run --profile work`
/** Script-path launcher of the same runtime. */
const GATEWAY_RUN_SCRIPT_CMDLINE = `"${PAYLOAD_PYTHON}" C:\\Hermes\\hermes-agent\\gateway\\run.py --profile work`

function reap(processes: RunningProcess[], overrides: Record<string, unknown> = {}) {
  const killProcess = vi.fn()

  const outcome = reapPackageRootedProcesses({
    installRoots: [ROOT],
    listProcesses: () => processes,
    killProcess,
    selfPid: 1,
    isWindows: true,
    ...overrides
  })

  return { outcome, killProcess }
}

function proc(
  pid: number,
  path: string | null,
  commandLine: string | null = null,
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

describe('reapPackageRootedProcesses', () => {
  it('kills a detached package-rooted daemon that a tree-kill cannot reach', () => {
    // The regression: gpg-agent reparented away from us, so it is in no tree we
    // own. Path scoping is what catches it.
    const { outcome, killProcess } = reap([proc(48236, GPG_AGENT, DAEMON_CMDLINE)])

    expect(outcome.matched).toBe(1)
    expect(outcome.killed).toEqual([48236])
    expect(killProcess).toHaveBeenCalledWith(48236)
  })

  it('leaves shared interpreters and entry points to their lifecycle owners', () => {
    const { outcome } = reap([
      proc(10, MAIN_EXE, `${MAIN_EXE} --app`),
      proc(11, PAYLOAD_PYTHON, 'idle-python'),
      proc(12, GPG_AGENT, DAEMON_CMDLINE)
    ])

    expect(outcome.matched).toBe(1)
    expect(outcome.killed).toEqual([12])
  })

  it('never kills processes outside the install root', () => {
    const { outcome, killProcess } = reap([
      proc(20, 'C:\\Program Files\\Git\\usr\\bin\\gpg-agent.exe', 'gpg-agent --daemon'),
      proc(21, 'C:\\Users\\arilo\\AppData\\Local\\Programs\\HermesBundled\\app\\Hermes.exe'),
      proc(22, GPG_AGENT, DAEMON_CMDLINE)
    ])

    // The user's OWN gpg-agent is exactly what a GNUPGHOME-scoped
    // `gpgconf --kill all` would have taken out. Path scoping leaves it alone.
    expect(outcome.killed).toEqual([22])
    expect(killProcess).not.toHaveBeenCalledWith(20)
    expect(killProcess).not.toHaveBeenCalledWith(21)
  })

  it('reaps a mutable install: pm-staged tooling under the managed store', () => {
    // No MSIX here. A daemonized node/git under <hermes root>\tools keeps its
    // image open, and on Windows a running image cannot be overwritten — so
    // the next `hermes update` fails to replace exactly those files.
    const { outcome, killProcess } = reap(
      [
        proc(70, STORE_NODE, `${STORE_NODE} daemon.js`),
        proc(71, `${TOOLS_ROOT}\\git-2.53.0+3-win32-arm64\\usr\\bin\\gpg-agent.exe`, DAEMON_CMDLINE),
        proc(72, 'C:\\Windows\\System32\\node.exe', 'node server.js')
      ],
      { installRoots: [null, TOOLS_ROOT] }
    )

    expect(outcome.killed).toEqual([71])
    expect(killProcess).not.toHaveBeenCalledWith(72)
  })

  it('reaps both roots in one pass when an install has both', () => {
    const { outcome } = reap(
      [proc(80, GPG_AGENT, DAEMON_CMDLINE), proc(81, STORE_NODE, `${STORE_NODE} daemon.js`)],
      { installRoots: [ROOT, TOOLS_ROOT] }
    )

    expect(outcome.matched).toBe(1)
    expect(outcome.killed).toEqual([80])
  })

  it('never reaps a live Hermes runtime process rooted in the payload', () => {
    // Bundled installs run the surviving gateway out of the payload, so its
    // image IS under the artifact root — but the gateway survives desktop
    // quit by contract. The `-m hermes_cli.main` invocation is what proves a
    // process is a legitimate Hermes surface rather than an orphan.
    const { outcome, killProcess } = reap([
      proc(90, PAYLOAD_PYTHON, GATEWAY_CMDLINE),
      proc(91, PAYLOAD_PYTHON, `"${PAYLOAD_PYTHON}" -m hermes_cli.main serve --port 8080`)
    ])

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('the ordinary-quit selector kills the daemon but not the gateway or shared-store consumers', () => {
    // Recorder for the wired quit call (roots = the artifact root only): a
    // store-python gateway session and a store-node process are unrelated
    // users of the machine-scoped store; the payload gateway survives the
    // quit. Only the daemonized tool is selected.
    const storePython = 'C:\\Users\\arilo\\AppData\\Local\\hermes\\tools\\python-3.11.16\\python.exe'

    const { outcome, killProcess } = reap(
      [
        proc(100, storePython, `"${storePython}" -m hermes_cli.main gateway run`),
        proc(101, STORE_NODE, `${STORE_NODE} script.js`),
        proc(102, PAYLOAD_PYTHON, GATEWAY_CMDLINE),
        proc(103, GPG_AGENT, `"${GPG_AGENT}" --use-standard-socket --daemon`)
      ],
      { installRoots: [ROOT] }
    )

    expect(outcome.killed).toEqual([103])
    expect(killProcess).toHaveBeenCalledTimes(1)
    expect(killProcess).toHaveBeenCalledWith(103)
  })

  it('protects the alternate runtime launcher shapes: -m gateway.run and gateway/run.py', () => {
    // The Scheduled-Task launcher, elevated-handoff respawn and older
    // releases do not all use `-m hermes_cli.main`; every launcher of the
    // live runtime is protected (C06-09 review).
    const { outcome, killProcess } = reap([
      proc(91, PAYLOAD_PYTHON, GATEWAY_RUN_MODULE_CMDLINE),
      proc(92, PAYLOAD_PYTHON, GATEWAY_RUN_SCRIPT_CMDLINE)
    ])

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('protects the live runtime DESCENDANTS (venv launcher / worker / tool chain)', () => {
    // gateway -> venv launcher shim -> worker python -> payload git: every
    // one is a legitimate user of the payload while its ancestor lives.
    const launcher = `${ROOT}\\app\\resources\\agent-payload\\venv\\Scripts\\python.exe`
    const worker = `${launcher} -c "import worker"`
    const git = `${ROOT}\\app\\resources\\agent-payload\\tools\\git-2.53.0+3-win32-arm64\\bin\\git.exe status`

    const { outcome, killProcess } = reap([
      proc(120, PAYLOAD_PYTHON, GATEWAY_CMDLINE),
      proc(121, launcher, worker, 120),
      proc(122, launcher, null, 121), // unreadable cmdline, but runtime-descended
      proc(123, `${ROOT}\\app\\resources\\agent-payload\\tools\\git\\bin\\git.exe`, git, 121)
    ])

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('protects flagged interpreters and children of unidentified processes', () => {
    const { outcome } = reap([
      proc(130, PAYLOAD_PYTHON, `"${PAYLOAD_PYTHON}" -u -X utf8 -m gateway.run`),
      proc(131, GPG_AGENT, DAEMON_CMDLINE, 130),
      proc(140, null, null),
      proc(141, GPG_AGENT, DAEMON_CMDLINE, 140),
      proc(150, GPG_AGENT, DAEMON_CMDLINE)
    ])

    expect(outcome.killed).toEqual([150])
  })

  it('conservatively skips a process whose command line cannot be read', () => {
    // An unreadable cmdline cannot prove the process is not a live runtime.
    // Missing one pinner reproduces a bug we already have; killing an
    // unidentified process is unbounded damage.
    const { outcome, killProcess } = reap([proc(40, PAYLOAD_PYTHON, null), proc(41, PAYLOAD_PYTHON, '')])

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('never kills itself', () => {
    const { outcome, killProcess } = reap([proc(99, MAIN_EXE, `${MAIN_EXE} --app`)], { selfPid: 99 })

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('skips pids the graceful backend teardown already owns', () => {
    const { outcome } = reap(
      [proc(30, MAIN_EXE, `${MAIN_EXE} --app`), proc(31, PAYLOAD_PYTHON, 'idle-python')],
      { excludePids: [30] }
    )

    expect(outcome.killed).toEqual([])
  })

  it('leaves a process whose path cannot be read alone', () => {
    // Missing one pinner reproduces a bug we already have; killing an
    // unidentified process is unbounded damage.
    const { outcome, killProcess } = reap([proc(40, null)])

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('keeps going when one kill throws, and reports it', () => {
    const killProcess = vi.fn((pid: number) => {
      if (pid === 50) {
        throw new Error('Access is denied')
      }
    })

    const outcome = reapPackageRootedProcesses({
      installRoots: [ROOT],
      listProcesses: () => [proc(50, GPG_AGENT, DAEMON_CMDLINE), proc(51, GPG_AGENT, DAEMON_CMDLINE)],
      killProcess,
      selfPid: 1,
      isWindows: true
    })

    expect(outcome.failed).toEqual([50])
    expect(outcome.killed).toEqual([51])
  })

  it('never lets a failed enumeration break quit', () => {
    const outcome = reapPackageRootedProcesses({
      installRoots: [ROOT],
      listProcesses: () => {
        throw new Error('enumeration failed')
      },
      killProcess: vi.fn(),
      selfPid: 1,
      isWindows: true
    })

    expect(outcome.skipped).toBe(true)
    expect(outcome.killed).toEqual([])
  })

  it('is a no-op on POSIX and when no install root resolves', () => {
    const killProcess = vi.fn()

    const posix = reapPackageRootedProcesses({
      installRoots: [ROOT],
      listProcesses: () => [proc(60, MAIN_EXE, `${MAIN_EXE} --app`)],
      killProcess,
      selfPid: 1,
      isWindows: false
    })

    const rootless = reapPackageRootedProcesses({
      installRoots: [null, ''],
      listProcesses: () => [proc(61, MAIN_EXE, `${MAIN_EXE} --app`)],
      killProcess,
      selfPid: 1,
      isWindows: true
    })

    expect(posix.skipped).toBe(true)
    expect(rootless.skipped).toBe(true)
    expect(killProcess).not.toHaveBeenCalled()
  })
})

describe('protectedRuntimePids', () => {
  it('extends protection transitively through parentage and stays empty with no runtime', () => {
    const running = [
      proc(10, PAYLOAD_PYTHON, GATEWAY_CMDLINE, 2),
      proc(11, PAYLOAD_PYTHON, 'worker', 10),
      proc(12, PAYLOAD_PYTHON, 'grandchild', 11),
      proc(13, GPG_AGENT, 'gpg-agent --daemon', null)
    ]

    expect(protectedRuntimePids(running)).toEqual(new Set([10, 11, 12]))
    expect(protectedRuntimePids([proc(14, GPG_AGENT, 'gpg-agent --daemon')])).toEqual(new Set())
  })
})

describe('listWindowsProcesses', () => {
  it('parses Win32_Process output (pid|parent|path|command line), mapping unreadable fields to null', () => {
    // The real shape: ExecutablePath / CommandLine are null for processes we
    // cannot open, so the script emits empty middle/tail fields.
    const stdout = [
      `48236|61728|${GPG_AGENT}|gpg-agent --daemon`,
      `22660|4|${MAIN_EXE}|`,
      '4|0||',
      ''
    ].join('\r\n')

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
