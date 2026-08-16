import { describe, expect, it, vi } from 'vitest'

import { forceDrainWindowsUpdateBlockers } from './windows-update-force-drain'

const blocked = (...pids: number[]) => ({
  kind: 'blocked' as const,
  result: {
    blocked: true,
    processes: pids.map(pid => ({
      pid,
      name: 'python.exe',
      cmdline: 'python.exe -m hermes_cli.main serve',
      forceDrainEligible: true
    }))
  }
})

const blockedWithEligibility = (...processes: Array<{ pid: number; forceDrainEligible?: boolean }>) => ({
  kind: 'blocked' as const,
  result: {
    blocked: true,
    processes: processes.map(({ pid, forceDrainEligible }) => ({
      pid,
      name: 'python.exe',
      cmdline: 'python.exe -m hermes_cli.main serve',
      forceDrainEligible
    }))
  }
})

describe('forceDrainWindowsUpdateBlockers', () => {
  it('does not kill or rescan a clear result', async () => {
    const scan = vi.fn()
    const forceKillProcessTree = vi.fn()
    const clear = { kind: 'clear' as const, result: { blocked: false, processes: [] } }

    await expect(forceDrainWindowsUpdateBlockers(clear, { forceKillProcessTree, scan })).resolves.toBe(clear)
    expect(forceKillProcessTree).not.toHaveBeenCalled()
    expect(scan).not.toHaveBeenCalled()
  })

  it('tree-kills each reported target PID once before confirming the scan is clear', async () => {
    const clear = { kind: 'clear' as const, result: { blocked: false, processes: [] } }
    const forceKillProcessTree = vi.fn()
    const scan = vi.fn().mockResolvedValue(clear)

    await expect(
      forceDrainWindowsUpdateBlockers(blocked(41, 42, 41), {
        forceKillProcessTree,
        scan,
        wait: async () => {}
      })
    ).resolves.toEqual(clear)

    expect(forceKillProcessTree).toHaveBeenCalledTimes(2)
    expect(forceKillProcessTree).toHaveBeenNthCalledWith(1, 41)
    expect(forceKillProcessTree).toHaveBeenNthCalledWith(2, 42)
    expect(scan).toHaveBeenCalledOnce()
  })

  it('leaves scanner-reported but non-eligible holders blocked without killing them', async () => {
    const outcome = blockedWithEligibility({ pid: 41, forceDrainEligible: false })
    const forceKillProcessTree = vi.fn()
    const scan = vi.fn()

    await expect(
      forceDrainWindowsUpdateBlockers(outcome, { forceKillProcessTree, scan, wait: async () => {} })
    ).resolves.toBe(outcome)

    expect(forceKillProcessTree).not.toHaveBeenCalled()
    expect(scan).not.toHaveBeenCalled()
  })

  it('keeps scanning after a tree-kill attempt throws', async () => {
    const clear = { kind: 'clear' as const, result: { blocked: false, processes: [] } }

    const forceKillProcessTree = vi.fn(() => {
      throw new Error('access denied')
    })

    const scan = vi.fn().mockResolvedValue(clear)

    await expect(
      forceDrainWindowsUpdateBlockers(blocked(41), { forceKillProcessTree, scan, wait: async () => {} })
    ).resolves.toEqual(clear)

    expect(scan).toHaveBeenCalledOnce()
  })

  it('returns a failed final rescan so the caller can fail closed', async () => {
    const probeFailure = { kind: 'probe-failure' as const, error: 'scanner unavailable' }
    const forceKillProcessTree = vi.fn()
    const scan = vi.fn().mockResolvedValue(probeFailure)

    await expect(
      forceDrainWindowsUpdateBlockers(blocked(41), { forceKillProcessTree, scan, wait: async () => {} })
    ).resolves.toEqual(probeFailure)

    expect(forceKillProcessTree).toHaveBeenCalledExactlyOnceWith(41)
    expect(scan).toHaveBeenCalledOnce()
  })

  it('returns a still-blocked result after its bounded force-drain passes', async () => {
    const forceKillProcessTree = vi.fn()
    const scan = vi.fn().mockResolvedValue(blocked(41))

    await expect(
      forceDrainWindowsUpdateBlockers(blocked(41), { forceKillProcessTree, scan, wait: async () => {} })
    ).resolves.toEqual(blocked(41))

    expect(forceKillProcessTree).toHaveBeenCalledTimes(3)
    expect(scan).toHaveBeenCalledTimes(3)
  })
})
