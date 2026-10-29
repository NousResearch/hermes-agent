import { act, render, renderHook, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'

// The poll fires getLocalHardware() every 5s while the item is shown; the
// backend spawns nvidia-smi per poll, so a hidden window must stop polling
// entirely and resume on visibilitychange (#120262).
const { getLocalHardware } = vi.hoisted(() => ({
  getLocalHardware: vi.fn().mockResolvedValue({
    gpu_name: 'NVIDIA Test GPU',
    gpu_util_percent: 10,
    vram_total_bytes: 8 << 30,
    vram_used_bytes: 1 << 30,
    vram_usable_bytes: 6 << 30,
    ram_total_bytes: 32 << 30,
    ram_available_bytes: 16 << 30,
    uma: false
  })
}))

vi.mock('@/hermes', () => ({ getLocalHardware }))

import { useSystemResourcesStatusbarItem } from './system-resources-statusbar'

describe('system-resources statusbar polling', () => {
  afterEach(() => {
    getLocalHardware.mockClear()
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

  it('aggregates every installed GPU in the chip and lists each in the menu', async () => {
    getLocalHardware.mockResolvedValue({
      gpu_name: 'NVIDIA RTX PRO 6000 A',
      gpu_util_percent: 1,
      vram_total_bytes: 96 * (1 << 30),
      vram_used_bytes: 1 << 30,
      vram_usable_bytes: 90 * (1 << 30),
      ram_total_bytes: 256 * (1 << 30),
      ram_available_bytes: 200 * (1 << 30),
      uma: false,
      gpus: [
        {
          index: 0,
          name: 'NVIDIA RTX PRO 6000 A',
          util_percent: 1,
          vram_used_bytes: 1 << 30,
          vram_total_bytes: 96 * (1 << 30),
          vram_free_bytes: 95 * (1 << 30)
        },
        {
          index: 1,
          name: 'NVIDIA RTX PRO 6000 B',
          util_percent: 40,
          vram_used_bytes: 8 * (1 << 30),
          vram_total_bytes: 96 * (1 << 30),
          vram_free_bytes: 88 * (1 << 30)
        }
      ]
    })

    $localModelsEnabled.set(true)
    $statusbarHiddenIds.set([])
    const { result, unmount } = renderHook(() => useSystemResourcesStatusbarItem())

    // Aggregate-only chip: mean(1, 40) = 21%, physical 9/192 GiB.
    await waitFor(() => expect(result.current.label).toBe('Σ GPU 21% · VRAM 9.0/192.0 GiB'))
    expect(result.current.title).toContain('arithmetic mean')

    const panel = render(<>{result.current.menuContent}</>)

    expect(panel.getByRole('region', { name: 'GPU 0 · NVIDIA RTX PRO 6000 A' })).toBeTruthy()
    expect(panel.getByRole('region', { name: 'GPU 1 · NVIDIA RTX PRO 6000 B' })).toBeTruthy()
    expect(panel.getByText('Σ All GPUs')).toBeTruthy()
    expect(within(panel.getByRole('region', { name: 'GPU 1 · NVIDIA RTX PRO 6000 B' })).getByText('40%')).toBeTruthy()
    unmount()
  })
})
