import { act, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'

// The poll fires getLocalHardware() every 5s while the item is shown; the
// backend spawns nvidia-smi per poll, so a hidden window must stop polling
// entirely and resume on visibilitychange (#120262).
const { getLocalHardware, getSystemStats } = vi.hoisted(() => ({
  getLocalHardware: vi.fn().mockResolvedValue({
    gpu_name: 'NVIDIA Test GPU',
    gpu_util_percent: 10,
    vram_total_bytes: 8 << 30,
    vram_used_bytes: 1 << 30,
    vram_usable_bytes: 6 << 30,
    ram_total_bytes: 32 << 30,
    ram_available_bytes: 16 << 30,
    uma: false
  }),
  // /api/system/stats. The hostname is only a mock value: remote routing comes
  // from getSystemStats going through hermesApi with no explicit connection,
  // exactly like getLocalHardware.
  getSystemStats: vi.fn().mockResolvedValue({
    hostname: 'remote-box',
    cpu_count: 8,
    psutil: true,
    cpu_percent: 12.4,
    memory: { total: 32 * 2 ** 30, available: 8 * 2 ** 30, used: 24 * 2 ** 30, percent: 75 },
    disk: { total: 200 * 2 ** 30, used: 150 * 2 ** 30, free: 50 * 2 ** 30, percent: 75 }
  })
}))

vi.mock('@/hermes', () => ({ getLocalHardware, getSystemStats }))

const NO_GPU_HARDWARE = {
  gpu_name: null,
  gpu_util_percent: null,
  vram_total_bytes: 0,
  vram_used_bytes: null,
  vram_usable_bytes: 0,
  vram_label: '',
  ram_total_bytes: 32 * 2 ** 30,
  ram_available_bytes: 16 * 2 ** 30,
  uma: false
}

// `n << 30` wraps past 2 GiB in JS (32-bit shift), so sizes here use `2 ** 30`.
const GPU_HARDWARE = {
  ...NO_GPU_HARDWARE,
  gpu_name: 'NVIDIA Test GPU',
  gpu_util_percent: 10,
  vram_total_bytes: 8 * 2 ** 30,
  vram_used_bytes: 1 * 2 ** 30,
  vram_usable_bytes: 6 * 2 ** 30
}

import { useSystemResourcesStatusbarItem } from './system-resources-statusbar'

describe('system-resources statusbar polling', () => {
  afterEach(() => {
    getLocalHardware.mockClear()
    getSystemStats.mockClear()
    vi.restoreAllMocks()
    $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
    $localModelsEnabled.set(true)
  })

  async function renderShown() {
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    // Mount poll, plus nothing scheduled while the window is visible-but-idle
    // beyond the immediate one (POLL_MS is 5s; the fake timers below avoid it).
    await waitFor(() => expect(getLocalHardware).toHaveBeenCalled())

    return { unmount }
  }

  it('does not poll while the document is hidden', async () => {
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)

    // Mounted while hidden: the mount poll bails out, so no call is expected —
    // give the effect a tick and assert the spawn never happened.
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])
    const { unmount } = renderHook(() => useSystemResourcesStatusbarItem())
    await act(async () => {})

    expect(getLocalHardware).not.toHaveBeenCalled()

    // Even the visibilitychange resume path must stay quiet while hidden.
    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(getLocalHardware).not.toHaveBeenCalled()
    unmount()
  })

  it('resumes polling when the window becomes visible again', async () => {
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)

    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])
    const { unmount } = renderHook(() => useSystemResourcesStatusbarItem())
    await act(async () => {})
    expect(getLocalHardware).not.toHaveBeenCalled()

    vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    await waitFor(() => expect(getLocalHardware).toHaveBeenCalledTimes(1))
    unmount()
  })

  it('stops polling and removes the listener on unmount', async () => {
    const { unmount } = await renderShown()
    getLocalHardware.mockClear()

    unmount()

    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(getLocalHardware).not.toHaveBeenCalled()
  })
})

describe('system-resources CPU and disk', () => {
  afterEach(() => {
    getLocalHardware.mockClear()
    getSystemStats.mockClear()
    $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
    $localModelsEnabled.set(true)
  })

  it('shows the backend host CPU, disk and hostname next to RAM', async () => {
    getLocalHardware.mockResolvedValueOnce(NO_GPU_HARDWARE)
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('CPU 12% · RAM 16G/32G'))
    expect(getSystemStats).toHaveBeenCalledTimes(1)

    render(<>{result.current.menuContent}</>)
    expect(screen.getByText('remote-box')).toBeTruthy()
    expect(screen.getByText('CPU')).toBeTruthy()
    expect(screen.getByText('12%')).toBeTruthy()
    expect(screen.getByText('Disk')).toBeTruthy()
    expect(screen.getByText('150.0 GB / 200.0 GB')).toBeTruthy()
    unmount()
  })

  it('shows the hostname as the header and the GPU name on its own line when a GPU is present', async () => {
    getLocalHardware.mockResolvedValueOnce(GPU_HARDWARE)
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('GPU 10% · 1G/8G'))
    await waitFor(() => expect(getSystemStats).toHaveBeenCalledTimes(1))

    const { unmount: unmountMenu } = render(<>{result.current.menuContent}</>)
    await waitFor(() => expect(screen.getByText('remote-box')).toBeTruthy())
    expect(screen.getByText('NVIDIA Test GPU')).toBeTruthy()
    expect(screen.getByText('CPU')).toBeTruthy()
    unmountMenu()
    unmount()
  })

  it('keeps CPU, disk and RAM (from the stats memory block) when the hardware route fails', async () => {
    getLocalHardware.mockRejectedValueOnce(new Error('500'))
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('CPU 12% · RAM 24G/32G'))

    render(<>{result.current.menuContent}</>)
    expect(screen.getByText('12%')).toBeTruthy()
    expect(screen.getByText('24.0 GB / 32.0 GB')).toBeTruthy()
    expect(screen.getByText('150.0 GB / 200.0 GB')).toBeTruthy()
    expect(screen.queryByText('GPU utilization')).toBeNull()
    unmount()
  })

  it('hides CPU and disk for a backend without psutil (a 200 with no psutil fields)', async () => {
    getLocalHardware.mockResolvedValueOnce(NO_GPU_HARDWARE)
    getSystemStats.mockResolvedValueOnce({ hostname: 'bare-box', cpu_count: 4, psutil: false })
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('RAM 16G/32G'))

    render(<>{result.current.menuContent}</>)
    expect(screen.getByText('bare-box')).toBeTruthy()
    expect(screen.queryByText('CPU')).toBeNull()
    expect(screen.queryByText('Disk')).toBeNull()
    unmount()
  })

  it('hides the disk row when the reported disk total is 0', async () => {
    getLocalHardware.mockResolvedValueOnce(NO_GPU_HARDWARE)
    getSystemStats.mockResolvedValueOnce({
      hostname: 'odd-box',
      cpu_count: 4,
      psutil: true,
      cpu_percent: 3,
      disk: { total: 0, used: 0, free: 0, percent: 0 }
    })
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('CPU 3% · RAM 16G/32G'))

    render(<>{result.current.menuContent}</>)
    expect(screen.getByText('CPU')).toBeTruthy()
    expect(screen.queryByText('Disk')).toBeNull()
    unmount()
  })

  it('hides CPU and disk when the backend has no stats route, keeping the hardware rows', async () => {
    getLocalHardware.mockResolvedValueOnce(GPU_HARDWARE)
    getSystemStats.mockRejectedValueOnce(new Error('404'))
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('GPU 10% · 1G/8G'))

    render(<>{result.current.menuContent}</>)
    expect(screen.getByText('NVIDIA Test GPU')).toBeTruthy()
    expect(screen.getByText('RAM')).toBeTruthy()
    expect(screen.queryByText('CPU')).toBeNull()
    expect(screen.queryByText('Disk')).toBeNull()
    unmount()
  })
})

describe('system-resources poll cadence', () => {
  afterEach(() => {
    $connection.set(null)
    vi.useRealTimers()
    getLocalHardware.mockClear()
    getSystemStats.mockClear()
    $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
    $localModelsEnabled.set(true)
  })

  // Older backends serve /api/system/stats as an async route that samples CPU
  // for 100ms on the event loop, so it runs every 15s, not every 5s tick.
  it('polls the stats route every 15s while hardware polls every 5s', async () => {
    vi.useFakeTimers()
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(getLocalHardware).toHaveBeenCalledTimes(1)
    expect(getSystemStats).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000)
    })
    expect(getLocalHardware).toHaveBeenCalledTimes(3)
    expect(getSystemStats).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(getLocalHardware).toHaveBeenCalledTimes(4)
    expect(getSystemStats).toHaveBeenCalledTimes(2)
    unmount()
  })

  // Switching backends re-reads both routes at once and drops the old
  // machine's values, instead of pairing a stale hostname with new numbers
  // until the next 15s stats tick.
  it('re-reads stats immediately and clears the old host on a connection switch', async () => {
    // One poll per connection: the initial one and the switch's re-read.
    getLocalHardware.mockResolvedValueOnce(NO_GPU_HARDWARE).mockResolvedValueOnce(NO_GPU_HARDWARE)
    $connection.set({ baseUrl: 'http://127.0.0.1:9119' } as never)
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    await waitFor(() => expect(result.current.label).toBe('CPU 12% · RAM 16G/32G'))
    expect(getSystemStats).toHaveBeenCalledTimes(1)

    let release: (value: unknown) => void = () => {}
    getSystemStats.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          release = resolve
        })
    )

    await act(async () => {
      // No connectionId on either side (local -> unregistered remote): the
      // baseUrl alone must restart the poll.
      $connection.set({ baseUrl: 'https://ssh-box.example:9119' } as never)
    })
    expect(getSystemStats).toHaveBeenCalledTimes(2)
    expect(result.current.label).not.toContain('CPU 12%')

    await act(async () => {
      release({ hostname: 'ssh-box', cpu_count: 2, psutil: true, cpu_percent: 80 })
    })
    await waitFor(() => expect(result.current.label).toBe('CPU 80% · RAM 16G/32G'))
    unmount()
  })

  it('does not start a second poll loop when the window becomes visible mid-poll', async () => {
    let release: (value: typeof NO_GPU_HARDWARE) => void = () => {}
    getLocalHardware.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          release = resolve
        })
    )
    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])

    const { unmount } = renderHook(() => useSystemResourcesStatusbarItem())
    await act(async () => {})
    expect(getLocalHardware).toHaveBeenCalledTimes(1)

    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })
    expect(getLocalHardware).toHaveBeenCalledTimes(1)

    await act(async () => {
      release(NO_GPU_HARDWARE)
    })
    unmount()
  })
})
