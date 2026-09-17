import type { execFile } from 'node:child_process'

import { describe, expect, it, vi } from 'vitest'

import { installPowerProtect } from './power-protect-setup'

describe('installPowerProtect', () => {
  it('does not invoke privileged commands outside macOS', async () => {
    const exec = vi.fn() as unknown as typeof execFile

    await expect(installPowerProtect(exec, 'linux')).resolves.toEqual({
      error: 'Power Protect setup is only available on macOS.',
      ok: false
    })
    expect(exec).not.toHaveBeenCalled()
  })

  it('uses the native admin prompt and verifies both exact pmset commands', async () => {
    const calls: string[][] = []

    const exec = vi.fn(
      (
        command: string,
        args: string[],
        _options: unknown,
        callback: (error: Error | null, stdout: string, stderr: string) => void
      ) => {
        calls.push([command, ...args])

        if (command === '/usr/bin/osascript') {
          callback(null, '', '')
        } else {
          callback(null, 'NOPASSWD: /usr/bin/pmset -a disablesleep 1\n/usr/bin/pmset -a disablesleep 0\n', '')
        }
      }
    ) as unknown as typeof execFile

    await expect(installPowerProtect(exec, 'darwin')).resolves.toEqual({ ok: true })
    expect(calls).toHaveLength(2)
    expect(calls[0]?.[0]).toBe('/usr/bin/osascript')
    expect(calls[0]?.[1]).toBe('-e')
    expect(calls[0]?.[2]).toContain('/usr/sbin/visudo -c -f')
    expect(calls[0]?.[2]).toContain('/etc/sudoers.d/hermes-power-protect')
    expect(calls[1]).toEqual(['/usr/bin/sudo', '-n', '-l'])
  })
})
