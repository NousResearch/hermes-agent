import { beforeEach, describe, expect, it, vi } from 'vitest'

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }))

vi.mock('@tauri-apps/api/core', () => ({ invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn() }))

import { $bootstrap, launchHermesDesktop } from '../apps/bootstrap-installer/src/store'

describe('launchHermesDesktop', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    invoke.mockReset()
    $bootstrap.set({
      status: 'completed',
      protocolVersion: null,
      stages: {},
      stageOrder: [],
      currentStage: null,
      installRoot: '/tmp/hermes',
      error: null,
      logs: [],
    })
  })

  it('rejects with a retryable error when the backend launch stalls', async () => {
    invoke.mockReturnValue(new Promise(() => {}))

    const launch = launchHermesDesktop()
    const rejection = expect(launch).rejects.toThrow(
      'The desktop launch request timed out. Please try Launch again.',
    )
    await vi.advanceTimersByTimeAsync(30_000)

    await rejection
  })

  it('resolves when the backend launch completes before the timeout', async () => {
    invoke.mockResolvedValue(undefined)

    await expect(launchHermesDesktop()).resolves.toBeUndefined()
    expect(invoke).toHaveBeenCalledWith('launch_hermes_desktop', { installRoot: '/tmp/hermes' })
  })
})
