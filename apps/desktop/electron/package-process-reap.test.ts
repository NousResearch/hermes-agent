import { describe, expect, it, vi } from 'vitest'

import {
  isUnderInstallRoot,
  listWindowsProcesses,
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

/** Same class, mutable install: pm-staged node holding its own image open. */
const STORE_NODE = `${TOOLS_ROOT}\\node-26.7.0-win32-arm64\\node.exe`

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
    const other = 'C:\\Program Files\\WindowsApps\\8bitSolutionsLLC.bitwardendesktop_2026.7.0.0_arm64__x\\app\\Bitwarden.exe'

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
    const { outcome, killProcess } = reap([{ pid: 48236, path: GPG_AGENT }])

    expect(outcome.matched).toBe(1)
    expect(outcome.killed).toEqual([48236])
    expect(killProcess).toHaveBeenCalledWith(48236)
  })

  it('reaps every package-rooted process, whatever the image name', () => {
    const { outcome } = reap([
      { pid: 10, path: MAIN_EXE },
      { pid: 11, path: PAYLOAD_PYTHON },
      { pid: 12, path: GPG_AGENT }
    ])

    expect(outcome.matched).toBe(3)
    expect(outcome.killed).toEqual([10, 11, 12])
  })

  it('never kills processes outside the install root', () => {
    const { outcome, killProcess } = reap([
      { pid: 20, path: 'C:\\Program Files\\Git\\usr\\bin\\gpg-agent.exe' },
      { pid: 21, path: 'C:\\Users\\arilo\\AppData\\Local\\Programs\\HermesBundled\\app\\Hermes.exe' },
      { pid: 22, path: GPG_AGENT }
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
        { pid: 70, path: STORE_NODE },
        { pid: 71, path: `${TOOLS_ROOT}\\git-2.53.0+3-win32-arm64\\usr\\bin\\gpg-agent.exe` },
        { pid: 72, path: 'C:\\Windows\\System32\\node.exe' }
      ],
      { installRoots: [null, TOOLS_ROOT] }
    )

    expect(outcome.killed).toEqual([70, 71])
    expect(killProcess).not.toHaveBeenCalledWith(72)
  })

  it('reaps both roots in one pass when an install has both', () => {
    const { outcome } = reap(
      [
        { pid: 80, path: GPG_AGENT },
        { pid: 81, path: STORE_NODE }
      ],
      { installRoots: [ROOT, TOOLS_ROOT] }
    )

    expect(outcome.matched).toBe(2)
    expect(outcome.killed).toEqual([80, 81])
  })

  it('never kills itself', () => {
    const { outcome, killProcess } = reap([{ pid: 99, path: MAIN_EXE }], { selfPid: 99 })

    expect(outcome.matched).toBe(0)
    expect(killProcess).not.toHaveBeenCalled()
  })

  it('skips pids the graceful backend teardown already owns', () => {
    const { outcome } = reap([
      { pid: 30, path: MAIN_EXE },
      { pid: 31, path: PAYLOAD_PYTHON }
    ], { excludePids: [30] })

    expect(outcome.killed).toEqual([31])
  })

  it('leaves a process whose path cannot be read alone', () => {
    // Missing one pinner reproduces a bug we already have; killing an
    // unidentified process is unbounded damage.
    const { outcome, killProcess } = reap([{ pid: 40, path: null }])

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
      listProcesses: () => [
        { pid: 50, path: MAIN_EXE },
        { pid: 51, path: GPG_AGENT }
      ],
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
      listProcesses: () => [{ pid: 60, path: MAIN_EXE }],
      killProcess,
      selfPid: 1,
      isWindows: false
    })

    const rootless = reapPackageRootedProcesses({
      installRoots: [null, ''],
      listProcesses: () => [{ pid: 61, path: MAIN_EXE }],
      killProcess,
      selfPid: 1,
      isWindows: true
    })

    expect(posix.skipped).toBe(true)
    expect(rootless.skipped).toBe(true)
    expect(killProcess).not.toHaveBeenCalled()
  })
})

describe('listWindowsProcesses', () => {
  it('parses Get-Process output, mapping an unreadable path to null', () => {
    // The real shape: `.Path` throws for processes we cannot open, so the
    // script emits an empty tail for those.
    const stdout = [
      `48236|${GPG_AGENT}`,
      `22660|${MAIN_EXE}`,
      '4|',
      ''
    ].join('\r\n')

    const parsed = listWindowsProcesses(() => stdout)

    expect(parsed).toEqual([
      { pid: 48236, path: GPG_AGENT },
      { pid: 22660, path: MAIN_EXE },
      { pid: 4, path: null }
    ])
  })

  it('keeps paths that contain spaces intact', () => {
    // "C:\Program Files\..." — splitting on the FIRST separator is what
    // preserves the rest of the path.
    const parsed = listWindowsProcesses(() => `10|${MAIN_EXE}`)

    expect(parsed[0].path).toBe(MAIN_EXE)
  })

  it('ignores malformed lines instead of inventing pids', () => {
    const parsed = listWindowsProcesses(() => ['garbage', '|no-pid', 'abc|path', ''].join('\n'))

    expect(parsed).toEqual([])
  })

  it('runs hidden and bounded so it cannot stall or flash a console on quit', () => {
    const calls: Array<{ file: string; options: { timeout: number; windowsHide: boolean } }> = []

    const execFile = (
      file: string,
      _args: string[],
      options: { timeout: number; windowsHide: boolean }
    ): string => {
      calls.push({ file, options })

      return ''
    }

    listWindowsProcesses(execFile)

    expect(calls[0].file).toBe('powershell.exe')
    expect(calls[0].options.windowsHide).toBe(true)
    expect(calls[0].options.timeout).toBeGreaterThan(0)
  })
})
