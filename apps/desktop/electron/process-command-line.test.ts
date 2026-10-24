import { describe, expect, it, vi } from 'vitest'

import { processCommandLine } from './process-command-line'

/**
 * The bug this module fixes: `isHermesProcess` fell back to POSIX `ps` when
 * /proc was missing, so on Windows the fallback always failed AND the shell's
 * "not recognized" error leaked into the user's console. These tests pin the
 * platform-native probe for each platform.
 */

describe('processCommandLine', () => {
  it('uses Get-CimInstance on Windows, never ps', () => {
    const execFileImpl = vi.fn(() => 'C:\\python.exe -m hermes_cli.main gateway run') as never

    const out = processCommandLine(4242, {
      platform: 'win32',
      isAlive: () => true,
      execFileImpl,
    })

    expect(out).toContain('hermes')
    expect(execFileImpl).toHaveBeenCalledTimes(1)

    const [command, args] = (execFileImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0]

    expect(command).toBe('powershell.exe')
    // The probe is ONE -Command argument; the whole command line must
    // contain the CIM query and the pid.
    expect(args.join(' ')).toContain('Get-CimInstance Win32_Process')
    expect(args.join(' ')).toContain('ProcessId = 4242')
    expect(args).toContain('-NoProfile')
  })

  it('captures stderr on Windows so a failed probe stays silent', () => {
    const execFileImpl = vi.fn(() => 'C:\\python.exe -m hermes_cli.main gateway run') as never

    processCommandLine(4242, { platform: 'win32', isAlive: () => true, execFileImpl })

    const options = (execFileImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0][2]

    // execFileSync's default sends the child's stderr to the parent's stderr
    // -- that is how "ps is not recognized" reached the user's console.
    expect(options.stdio).toEqual(['ignore', 'pipe', 'pipe'])
  })

  it('reads /proc on Linux without spawning anything', () => {
    const execFileImpl = vi.fn(() => 'should not be called') as never
    const readFileSync = vi.fn(() => 'python3\0-m\0hermes_cli\0gateway\0') as never

    const out = processCommandLine(77, {
      platform: 'linux',
      isAlive: () => true,
      readFileSync,
      execFileImpl,
    })

    expect(out).toContain('hermes_cli')
    expect(execFileImpl).not.toHaveBeenCalled()
  })

  it('falls back to ps on macOS, where /proc does not exist', () => {
    const execFileImpl = vi.fn(() => '/usr/bin/python3 -m hermes_cli.main gateway run') as never
    const readFileSync = vi.fn(() => {
      throw new Error('ENOENT')
    }) as never

    const out = processCommandLine(77, {
      platform: 'darwin',
      isAlive: () => true,
      readFileSync,
      execFileImpl,
    })

    expect(out).toContain('hermes')
    const [command, args] = (execFileImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0]

    expect(command).toBe('ps')
    expect(args).toEqual(['-p', '77', '-o', 'args='])
  })

  it('returns null for a dead pid without probing', () => {
    const execFileImpl = vi.fn(() => 'x') as never

    expect(processCommandLine(99, { platform: 'win32', isAlive: () => false, execFileImpl })).toBeNull()
    expect(execFileImpl).not.toHaveBeenCalled()
  })

  it('returns null when the probe fails -- unknown, not a false answer', () => {
    const execFileImpl = vi.fn(() => {
      throw new Error('exit 1')
    }) as never

    expect(processCommandLine(99, { platform: 'win32', isAlive: () => true, execFileImpl })).toBeNull()
  })

  it('rejects a nonsensical pid without probing', () => {
    const execFileImpl = vi.fn(() => 'x') as never

    expect(processCommandLine(0, { platform: 'win32', isAlive: () => true, execFileImpl })).toBeNull()
    expect(processCommandLine(-1, { platform: 'win32', isAlive: () => true, execFileImpl })).toBeNull()
    expect(execFileImpl).not.toHaveBeenCalled()
  })
})
