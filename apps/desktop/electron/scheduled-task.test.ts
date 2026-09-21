import { describe, expect, it } from 'vitest'

import {
  buildScheduledTaskSpec,
  classifyTaskState,
  isOurActionShape,
  isTaskBranchSafe,
  isTaskPathSafe,
  parseTaskActionFromXml,
  taskActionEquals,
  UNATTENDED_TASK_NAME,
  UNATTENDED_TASK_STAMP_FILENAME
} from './scheduled-task'
import type { ScheduledTaskSpec } from './scheduled-task'

const SCRIPT_PATH = 'C:\\Users\\joe\\Hermes\\hermes-agent\\scripts\\desktop-update\\windows.ps1'
const INSTALL_ROOT = 'C:\\Users\\joe\\Hermes\\hermes-agent'
const RELAUNCH_EXE = 'C:\\Users\\joe\\AppData\\Local\\Programs\\hermes\\Hermes.exe'
const BINARY_PATH = 'C:\\Users\\joe\\Hermes\\hermes-setup.exe'

function scriptSpec(overrides: Partial<Parameters<typeof buildScheduledTaskSpec>[0]> = {}): ScheduledTaskSpec {
  return buildScheduledTaskSpec({
    scriptPath: SCRIPT_PATH,
    installRoot: INSTALL_ROOT,
    relaunchExe: RELAUNCH_EXE,
    branch: 'main',
    hour: 2,
    minute: 0,
    ...overrides
  })
}

describe('isTaskPathSafe (against command-line injection)', () => {
  it('accepts ordinary Windows paths with spaces, dots, parens and non-ASCII', () => {
    expect(isTaskPathSafe('C:\\Users\\José\\Hermes\\hermes-agent')).toBe(true)
    expect(isTaskPathSafe('C:\\Program Files (x86)\\hermes\\hermes-agent')).toBe(true)
    expect(isTaskPathSafe('C:\\Users\\joe\\AppData\\Local\\Programs\\hermes\\Hermes.exe')).toBe(true)
    expect(isTaskPathSafe('D:\\')).toBe(false) // trailing separator — see below
    expect(isTaskPathSafe('C:\\Users\\joe\\hermes agent\\checkout')).toBe(true)
  })

  it('rejects every cmd/CreateProcess metacharacter even inside a would-be quote', () => {
    for (const ch of ['"', '&', '|', '<', '>', '^', '%', '!', '`']) {
      expect(isTaskPathSafe(`C:\\Users\\joe\\am${ch}bad`), `char ${ch}`).toBe(false)
      expect(isTaskPathSafe(`C:\\Users\\joe\\${ch}`), `char ${ch} at end`).toBe(false)
    }
  })

  it('rejects control characters, tab/newline padding and whitespace tricks', () => {
    expect(isTaskPathSafe('C:\\Users\\joe\\a\u0000b')).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe\\a\tb')).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe\\a\nb')).toBe(false)
    expect(isTaskPathSafe(' C:\\Users\\joe')).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe ')).toBe(false)
  })

  it('rejects empty, over-long and trailing-separator paths (quote-escape hazard)', () => {
    expect(isTaskPathSafe('')).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe\\' + 'x'.repeat(300))).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe\\')).toBe(false)
    expect(isTaskPathSafe('C:\\Users\\joe/')).toBe(false)
    expect(isTaskPathSafe(null as unknown as string)).toBe(false)
    expect(isTaskPathSafe(undefined as unknown as string)).toBe(false)
  })
})

describe('isTaskBranchSafe', () => {
  it('accepts stock refs', () => {
    expect(isTaskBranchSafe('main')).toBe(true)
    expect(isTaskBranchSafe('feat/desktop-unattended-updates')).toBe(true)
    expect(isTaskBranchSafe('release/2026.09')).toBe(true)
    expect(isTaskBranchSafe('bb/gui')).toBe(true)
    expect(isTaskBranchSafe('a_b-c.d/e')).toBe(true)
  })

  it('rejects injection-shaped and dummy refs', () => {
    expect(isTaskBranchSafe('')).toBe(false)
    expect(isTaskBranchSafe('main; calc.exe')).toBe(false)
    expect(isTaskBranchSafe('main & whoami')).toBe(false)
    expect(isTaskBranchSafe('main"&&"whoami')).toBe(false)
    expect(isTaskBranchSafe('-main')).toBe(false) // leading dash parses as a switch
    expect(isTaskBranchSafe('main --flag')).toBe(false)
    expect(isTaskBranchSafe('main\n')).toBe(false)
    expect(isTaskBranchSafe('m'.repeat(201))).toBe(false)
    expect(isTaskBranchSafe(null as unknown as string)).toBe(false)
  })
})

describe('buildScheduledTaskSpec (deterministic action)', () => {
  it('produces the exact canned script hand-off action', () => {
    const spec = scriptSpec()

    expect(spec.taskName).toBe(UNATTENDED_TASK_NAME)
    expect(spec.handoff).toBe('script')
    expect(spec.startClock).toBe('02:00')
    expect(spec.action).toBe(
      'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden ' +
        `-File "${SCRIPT_PATH}" -InstallRoot "${INSTALL_ROOT}" -Branch main -DesktopPid 0 ` +
        `-RelaunchExe "${RELAUNCH_EXE}" -Unattended -NoUi`
    )
  })

  it('is a pure function: identical inputs, identical action', () => {
    const a = scriptSpec()
    const b = scriptSpec()

    expect(a.action).toBe(b.action)
    expect(a.schtasksArgs).toEqual(b.schtasksArgs)
  })

  it('zero-pads the schedule into HH:MM across the day boundary', () => {
    expect(scriptSpec({ hour: 2, minute: 0 }).startClock).toBe('02:00')
    expect(scriptSpec({ hour: 23, minute: 59 }).startClock).toBe('23:59')
    expect(scriptSpec({ hour: 0, minute: 5 }).startClock).toBe('00:05')
    expect(scriptSpec({ hour: 12, minute: 0 }).startClock).toBe('12:00')
  })

  it('builds the binary hand-off fallback when only the staged exe exists', () => {
    const spec = buildScheduledTaskSpec({
      updaterBinaryPath: BINARY_PATH,
      installRoot: INSTALL_ROOT,
      relaunchExe: RELAUNCH_EXE,
      branch: 'bb/gui',
      hour: 23,
      minute: 59
    })

    expect(spec.handoff).toBe('binary')
    expect(spec.action).toBe(`"${BINARY_PATH}" --update --branch bb/gui`)
    expect(spec.startClock).toBe('23:59')
  })

  it('fails CLOSED on unsafe inputs and on missing/duplicate handoffs', () => {
    expect(() =>
      buildScheduledTaskSpec({
        installRoot: INSTALL_ROOT,
        relaunchExe: RELAUNCH_EXE,
        branch: 'main',
        hour: 2,
        minute: 0
      })
    ).toThrow(/exactly one/)

    expect(() =>
      scriptSpec({
        scriptPath: 'C:\\Users\\joe\\x" & calc.exe &"\\windows.ps1',
        updaterBinaryPath: BINARY_PATH
      })
    ).toThrow(/exactly one/)

    expect(() => scriptSpec({ relaunchExe: `${RELAUNCH_EXE} & calc` })).toThrow(/unsafe relaunch/)
    expect(() => scriptSpec({ installRoot: 'C:\\Users\\joe\\x%TEMP%\\y' })).toThrow(/unsafe install/)
    expect(() => scriptSpec({ branch: 'main; calc' })).toThrow(/unsafe branch/)
    expect(() => scriptSpec({ hour: 24, minute: 0 })).toThrow(/invalid/)
    expect(() => scriptSpec({ hour: 2, minute: 61 })).toThrow(/invalid/)
  })

  it('registers per-user and LIMITED: no /ru, no /rp, /rl LIMITED, daily at the slot', () => {
    const spec = scriptSpec({ hour: 2, minute: 0 })
    const args = spec.schtasksArgs

    expect(args).toEqual([
      '/create',
      '/tn',
      UNATTENDED_TASK_NAME,
      '/tr',
      `"${spec.action}"`,
      '/sc',
      'daily',
      '/st',
      '02:00',
      '/rl',
      'LIMITED',
      '/f'
    ])
    // Least privilege: the task must be owned by the CURRENT user only —
    // any /ru or /rp would introduce a stored credential or a different
    // principal. Assert they can never appear.
    expect(args).not.toContain('/ru')
    expect(args).not.toContain('/rp')
    expect(args).not.toContain('/ru ')
  })

  it('uninstall and read-back argv are the exact named task only', () => {
    const spec = scriptSpec()

    expect(spec.schtasksDeleteArgs).toEqual(['/delete', '/tn', UNATTENDED_TASK_NAME, '/f'])
    expect(spec.schtasksQueryArgs).toEqual(['/query', '/tn', UNATTENDED_TASK_NAME, '/xml'])
  })

  it('stamp filename is shared contract with windows.ps1', () => {
    expect(UNATTENDED_TASK_STAMP_FILENAME).toBe('.hermes-unattended-last-run')
  })
})

describe('parseTaskActionFromXml (read-back verification)', () => {
  it('parses a UTF-8 schtasks XML export', () => {
    const xml =
      '<?xml version="1.0" encoding="UTF-16"?><Task version="1.2"><Actions Context="Author"><Exec><Command>powershell.exe</Command><Arguments>-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' +
      SCRIPT_PATH +
      '" -InstallRoot "' +
      INSTALL_ROOT +
      '" -Branch main -DesktopPid 0 -RelaunchExe "' +
      RELAUNCH_EXE +
      '" -Unattended -NoUi</Arguments></Exec></Actions></Task>'

    const parsed = parseTaskActionFromXml(xml)

    expect(parsed).not.toBeNull()
    expect(parsed?.command).toBe('powershell.exe')
    expect(parsed?.arguments).toContain(`-File "${SCRIPT_PATH}"`)
  })

  it('decodes the UTF-16LE BOM output real schtasks emits', () => {
    const xml =
      '<Task><Actions><Exec><Command>powershell.exe</Command><Arguments>-Branch main -Unattended</Arguments></Exec></Actions></Task>'
    const utf16le = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from(xml, 'utf16le')])

    const parsed = parseTaskActionFromXml(utf16le)

    expect(parsed?.command).toBe('powershell.exe')
    expect(parsed?.arguments).toBe('-Branch main -Unattended')
  })

  it('decodes XML entities in the read-back', () => {
    const xml =
      '<Task><Actions><Exec><Command>powershell.exe</Command><Arguments>-File &quot;C:\\a b\\w.ps1&quot; -Branch main</Arguments></Exec></Actions></Task>'
    const parsed = parseTaskActionFromXml(xml)

    expect(parsed?.arguments).toBe('-File "C:\\a b\\w.ps1" -Branch main')
  })

  it('returns null on garbage and on documents without an Exec action', () => {
    expect(parseTaskActionFromXml('')).toBeNull()
    expect(parseTaskActionFromXml('<Task><Settings/></Task>')).toBeNull()
    expect(parseTaskActionFromXml('ERROR: The system cannot find the file specified.')).toBeNull()
    expect(parseTaskActionFromXml(Buffer.from('nonsense bytes'))).toBeNull()
  })
})

describe('taskActionEquals / classifyTaskState (ownership)', () => {
  const spec = scriptSpec()

  const parsedFrom = (command: string, args: string) => ({ command, arguments: args })

  it('accepts our exact action (whitespace-collapsed)', () => {
    const parsed = parsedFrom(
      'powershell.exe',
      `-NoProfile  -ExecutionPolicy Bypass -WindowStyle Hidden -File "${SCRIPT_PATH}" -InstallRoot "${INSTALL_ROOT}" -Branch main -DesktopPid 0 -RelaunchExe "${RELAUNCH_EXE}" -Unattended -NoUi`
    )

    expect(taskActionEquals(parsed, spec)).toBe(true)
    expect(classifyTaskState(parsed, spec)).toEqual({ kind: 'installed' })
  })

  it('accepts a quoted <Command> token (Task Scheduler echo form)', () => {
    const parsed = parsedFrom(`"${BINARY_PATH}"`, `--update --branch main`)
    const binarySpec = buildScheduledTaskSpec({
      updaterBinaryPath: BINARY_PATH,
      installRoot: INSTALL_ROOT,
      relaunchExe: RELAUNCH_EXE,
      branch: 'main',
      hour: 2,
      minute: 0
    })

    expect(taskActionEquals(parsed, binarySpec)).toBe(true)
  })

  it('rejects a tampered action (injected command after our flags)', () => {
    const tampered = parsedFrom(
      'powershell.exe',
      `-File "${SCRIPT_PATH}" -Branch main -UserProfile` // unknown switch, ours would end -NoUi
    )

    expect(taskActionEquals(tampered, spec)).toBe(false)
    expect(classifyTaskState(tampered, spec)).toEqual({
      kind: 'foreign',
      message: 'a task with this name is not ours'
    })
  })

  it('rejects an unrelated executable squatting on our task name', () => {
    const squatter = parsedFrom('C:\\Windows\\System32\\calc.exe', '/c calc')

    expect(classifyTaskState(squatter, spec).kind).toBe('foreign')
  })

  it('treats an unreadable/missing Exec as foreign (never touch it)', () => {
    expect(classifyTaskState(null, spec).kind).toBe('foreign')
  })
})

describe('isOurActionShape (structural, path-independent)', () => {
  it('recognizes our script shape with different (moved) install paths', () => {
    expect(
      isOurActionShape(
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden ' +
          '-File "D:\\moved\\hermes-agent\\scripts\\desktop-update\\windows.ps1" ' +
          '-InstallRoot "D:\\moved\\hermes-agent" -Branch feat/x -DesktopPid 0 ' +
          '-RelaunchExe "D:\\moved\\Hermes.exe" -Unattended -NoUi'
      )
    ).toBe(true)
  })

  it('recognizes our binary shape', () => {
    expect(isOurActionShape('"C:\\hermes\\hermes-setup.exe" --update --branch main')).toBe(true)
  })

  it('rejects foreign and injection-shaped actions', () => {
    expect(isOurActionShape('calc.exe /c calc')).toBe(false)
    expect(isOurActionShape('powershell.exe -File "C:\\w.ps1" -Branch main & whoami')).toBe(false)
    expect(isOurActionShape('cmd.exe /c del C:\\* /q')).toBe(false)
    expect(isOurActionShape('')).toBe(false)
    // Our flag sequence but an injected trailing command after the fixed tail.
    expect(
      isOurActionShape(
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden ' +
          '-File "C:\\a\\windows.ps1" -InstallRoot "C:\\a" -Branch main -DesktopPid 0 ' +
          '-RelaunchExe "C:\\a\\Hermes.exe" -Unattended -NoUi & calc'
      )
    ).toBe(false)
    expect(isOurActionShape(null as unknown as string)).toBe(false)
  })
})
