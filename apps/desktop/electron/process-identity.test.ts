import type * as ChildProcess from 'node:child_process'

import { describe, expect, it, vi } from 'vitest'

import { isHermesProcess } from './process-identity'

const probe = vi.hoisted(() => ({
  calls: [] as { command: string, args: string[] }[],
  stub: null as null | ((command: string, args: string[]) => string)
}))

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcess>()

  const intercept = (name: 'execFileSync' | 'execSync') => {
    const real = actual[name] as any

    return (command: string, args: string[], options: any) => {
      if (!probe.stub) {
        return real(command, args, options)
      }

      probe.calls.push({ command, args })

      return probe.stub(command, args)
    }
  }

  const execFileSync = intercept('execFileSync')
  const execSync = intercept('execSync')

  return { ...actual, default: { ...actual, execFileSync, execSync }, execFileSync, execSync }
})

/** The vitest process itself is the one PID a test can rely on being alive. */
const SELF = process.pid

/** A PID far beyond any the OS will hand out, so the liveness gate rejects it. */
const DEAD_PID = 2_147_483_646

/** What `Get-CimInstance Win32_Process.CommandLine` reports for a local gateway. */
const HERMES_GATEWAY_CMDLINE =
  '"C:\\Users\\u\\AppData\\Local\\hermes\\tools\\python-3.14.7\\python.exe" -m hermes_cli.main gateway run'

/**
 * Run `body` with `process.platform` stubbed and command probes intercepted.
 *
 * Platform handlers are stubbed rather than run because a macOS `ps` and a
 * Windows `powershell.exe` do not exist on the same host, and every CI lane
 * must exercise every branch.
 */
function onPlatform(
  platform: NodeJS.Platform,
  stub: (command: string, args: string[]) => string,
  body: () => void
) {
  const descriptor = Object.getOwnPropertyDescriptor(process, 'platform')!

  Object.defineProperty(process, 'platform', { ...descriptor, value: platform })
  probe.calls = []
  probe.stub = stub

  try {
    body()
  } finally {
    probe.stub = null
    Object.defineProperty(process, 'platform', descriptor)
  }
}

/**
 * A real Windows host: no /proc, and the procps `ps` the app ships rejects the
 * BSD-style `-o` that macOS accepts (`ps: unknown option -- o`). A probe that
 * lets Windows fall through to `ps` is therefore not modelling the platform —
 * it is asserting a behaviour Windows cannot deliver.
 */
function windowsHost(command: string, args: string[]): string {
  if (command === 'ps' && args.includes('-o')) {
    throw new Error('ps: unknown option -- o')
  }

  if (command === 'powershell.exe') {
    return HERMES_GATEWAY_CMDLINE
  }

  throw new Error(`unexpected probe: ${command} ${args.join(' ')}`)
}

describe('isHermesProcess', () => {
  it('never claims a PID that is not running, on any platform', () => {
    onPlatform('win32', windowsHost, () => {
      expect(isHermesProcess(DEAD_PID)).toBe(false)
      expect(probe.calls).toEqual([])
    })
  })

  it('identifies a hermes process on darwin via ps -o args=', () => {
    onPlatform(
      'darwin',
      (command, args) => {
        expect(command).toBe('ps')
        expect(args).toEqual(['-p', String(SELF), '-o', 'args='])

        return HERMES_GATEWAY_CMDLINE
      },
      () => {
        expect(isHermesProcess(SELF)).toBe(true)
      }
    )
  })

  it('does not claim a non-hermes process on darwin', () => {
    onPlatform('darwin', () => '"C:\\Windows\\System32\\notepad.exe"', () => {
      expect(isHermesProcess(SELF)).toBe(false)
    })
  })

  it('identifies a hermes process on win32 without delegating to ps', () => {
    onPlatform('win32', windowsHost, () => {
      expect(isHermesProcess(SELF)).toBe(true)

      // The regression: `ps` cannot answer on Windows, so a probe that reaches
      // it always throws and the identity check collapses to `false`.
      expect(probe.calls.map(call => call.command)).toEqual(['powershell.exe'])
    })
  })

  it('does not claim a non-hermes process on win32', () => {
    onPlatform(
      'win32',
      command => {
        if (command !== 'powershell.exe') {
          throw new Error('ps: unknown option -- o')
        }

        return '"C:\\Program Files\\Notepad++\\notepad++.exe"'
      },
      () => {
        expect(isHermesProcess(SELF)).toBe(false)
      }
    )
  })

  it('treats an empty CommandLine as not-hermes', () => {
    onPlatform('win32', () => '', () => {
      expect(isHermesProcess(SELF)).toBe(false)
    })
  })

  it('degrades to false when the probe itself fails', () => {
    onPlatform('win32', () => {
      throw new Error('powershell.exe is unavailable')
    }, () => {
      expect(isHermesProcess(SELF)).toBe(false)
    })
  })
})
