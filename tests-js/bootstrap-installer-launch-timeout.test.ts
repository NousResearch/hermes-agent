/// <reference types="vite/client" />
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }))

vi.mock('@tauri-apps/api/core', () => ({ invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn() }))

import { $bootstrap, launchHermesDesktop } from '../apps/bootstrap-installer/src/store'

const LAUNCH_BACKSTOP_MS = 35_000

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
      logs: []
    })
  })

  it('rejects with a retryable error when the backend launch stalls', async () => {
    // The never-settling invoke from issue #128804: the Launch button held
    // its spinner forever with no inline error.
    invoke.mockReturnValue(new Promise(() => {}))

    const launch = launchHermesDesktop()

    const rejection = expect(launch).rejects.toThrow(
      'The desktop launch request timed out. Please try Launch again.'
    )

    await vi.advanceTimersByTimeAsync(LAUNCH_BACKSTOP_MS)

    await rejection
  })

  it('does not fire the backstop while the backend is still inside its own deadline', async () => {
    // The backend bounds its own exe probe at 30s and returns a SPECIFIC
    // error (AV-hold, "run `hermes desktop`"). The renderer backstop must
    // sit above that deadline so the specific message wins the race.
    invoke.mockReturnValue(new Promise(() => {}))

    const launch = launchHermesDesktop()
    const assertion = expect(launch).rejects.toThrow('timed out')
    await vi.advanceTimersByTimeAsync(30_000)

    await expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(LAUNCH_BACKSTOP_MS - 30_000)

    await assertion
  })

  it('resolves when the backend launch completes before the timeout', async () => {
    invoke.mockResolvedValue(undefined)

    await expect(launchHermesDesktop()).resolves.toBeUndefined()
    expect(invoke).toHaveBeenCalledWith('launch_hermes_desktop', { installRoot: '/tmp/hermes' })
  })

  it('surfaces the backend rejection verbatim (specific error wins)', async () => {
    // A backend error must reach the success screen exactly as written:
    // the AV-hold copy names the real cause and the terminal fallback.
    invoke.mockRejectedValue(
      new Error(
        'Timed out looking for the Hermes desktop under /tmp/hermes. Antivirus software can hold a freshly built binary; retry, or start it with `hermes desktop` from a terminal.'
      )
    )

    await expect(launchHermesDesktop()).rejects.toThrow('Antivirus software')
  })

  it('does not blow up as an unhandled rejection when the invoke settles after the backstop', async () => {
    // The losing arm: the backstop fires first, and the (merely slow) invoke
    // settles afterwards. Promise.race already handles that late rejection;
    // an unhandled one would take the webview down.
    let settle: ((value: unknown) => void) | undefined
    invoke.mockReturnValue(
      new Promise((_resolve, reject) => {
        settle = reject
      })
    )

    const launch = launchHermesDesktop()

    const rejection = expect(launch).rejects.toThrow(
      'The desktop launch request timed out. Please try Launch again.'
    )

    await vi.advanceTimersByTimeAsync(LAUNCH_BACKSTOP_MS)
    settle?.(new Error('late backend rejection'))

    await rejection
  })
})
