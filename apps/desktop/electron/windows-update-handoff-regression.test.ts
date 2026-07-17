import { describe, expect, it } from 'vitest'

import { buildUpdateLaunchScript, schtasksCreateArgs } from './windows-update-launch'
import { collectProcessTreePids } from './windows-update-runtime'

describe('Windows one-click update handoff regression', () => {
  it('launches through a hidden scheduled task and waits for an updater ACK', () => {
    const script = buildUpdateLaunchScript({
      updaterPath: 'C:\\Hermes\\hermes-setup.exe',
      updaterArgs: ['--update', '--branch', 'main'],
      hermesHome: 'C:\\Hermes',
      pathEnv: 'C:\\Hermes\\venv\\Scripts',
      ackPath: 'C:\\Hermes\\logs\\ack.json',
      logPath: 'C:\\Hermes\\logs\\launch.log',
      requestId: 'request-1'
    })

    expect(schtasksCreateArgs('Hermes_UpdateLaunch', 'C:\\Hermes\\update-launch.ps1', '12:34').join(' '))
      .toContain('-WindowStyle Hidden')
    expect(script).toContain('Set-Content -LiteralPath $ack')
    expect(script).toContain('Wait-Process -Id $p.Id')
  })

  it('never includes the desktop itself when a recycled parent PID creates a false cycle', () => {
    const processes = [
      {
        pid: 200,
        parentPid: 100,
        creationTimeMs: 2_000,
        executablePath: 'C:\\Hermes\\venv\\Scripts\\hermes.exe',
        commandLine: ''
      },
      {
        pid: 100,
        parentPid: 200,
        creationTimeMs: 1_000,
        executablePath: 'C:\\Hermes\\Hermes.exe',
        commandLine: ''
      }
    ]

    expect(
      collectProcessTreePids(processes, 200, { excludePids: [100], expectedRootCreationTimeMs: 2_000 })
    ).toEqual([200])
  })
})
