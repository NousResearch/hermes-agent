import { render, renderHook, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import type { LocalHardware } from '@/types/hermes'

const { getLocalHardware } = vi.hoisted(() => ({ getLocalHardware: vi.fn() }))
vi.mock('@/hermes', () => ({ getLocalHardware }))
import { useSystemResourcesStatusbarItem } from './system-resources-statusbar'

const GiB = 2 ** 30

const base: LocalHardware = {
  gpu_name: 'Same GPU',
  gpu_util_percent: 10,
  vram_total_bytes: 32 * GiB,
  vram_used_bytes: 8 * GiB,
  vram_usable_bytes: 30 * GiB,
  vram_label: '32 GB',
  ram_total_bytes: 128 * GiB,
  ram_available_bytes: 64 * GiB,
  uma: false
}

afterEach(() => {
  vi.restoreAllMocks()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

async function shown(hardware: LocalHardware) {
  getLocalHardware.mockResolvedValue(hardware)
  $localModelsEnabled.set(true)
  $statusbarHiddenIds.set([])
  vi.spyOn(globalThis.document, 'hidden', 'get').mockReturnValue(false)
  const hook = renderHook(() => useSystemResourcesStatusbarItem())
  await waitFor(() => expect(String(hook.result.current.label)).toContain('GiB'))

  return hook
}

it('shows both duplicate-name GPUs, physical total, and RAM in the actual menu', async () => {
  const hook = await shown({
    ...base,
    gpus: [
      { index: 0, name: 'Same GPU', util_percent: 10, vram_used_bytes: 8 * GiB, vram_total_bytes: 32 * GiB },
      { index: 1, name: 'Same GPU', util_percent: 90, vram_used_bytes: 24 * GiB, vram_total_bytes: 32 * GiB }
    ]
  })

  expect(hook.result.current.label).toBe('Σ GPU 50% · VRAM 32.0/64.0 GiB')
  expect(hook.result.current.title).toContain('arithmetic mean')
  expect(hook.result.current.title).toContain('not pooled inference capacity')
  const menu = render(<>{hook.result.current.menuContent}</>)
  expect(menu.getByText('GPU 0 · Same GPU')).toBeTruthy()
  expect(menu.getByText('GPU 1 · Same GPU')).toBeTruthy()
  expect(menu.getByText('64.0 GiB / 128.0 GiB')).toBeTruthy()
  expect(menu.getByText('32.0 GiB / 64.0 GiB')).toBeTruthy()

  for (const [index, util, used] of [
    [0, 10, 8],
    [1, 90, 24]
  ]) {
    const section = within(menu.getByRole('region', { name: `GPU ${index} · Same GPU` }))
    expect(section.getByRole('heading', { name: `GPU ${index} · Same GPU` })).toBeTruthy()
    expect(section.getByText(`${util}%`)).toBeTruthy()
    expect(section.getByText(`${used}.0 GiB / 32.0 GiB`)).toBeTruthy()
  }

  hook.unmount()
})

it('preserves zero, marks unavailable readings and incomplete sums unknown', async () => {
  const hook = await shown({
    ...base,
    gpus: [
      { index: 0, name: 'A', util_percent: 0, vram_used_bytes: 0, vram_total_bytes: 32 * GiB },
      { index: 1, name: 'B', util_percent: null, vram_used_bytes: null, vram_total_bytes: null }
    ]
  })

  expect(hook.result.current.label).toBe('Σ GPU — · VRAM —/— GiB')
  const menu = render(<>{hook.result.current.menuContent}</>)
  const zero = within(menu.getByRole('region', { name: 'GPU 0 · A' }))
  expect(zero.getByText('0%')).toBeTruthy()
  expect(zero.getByText('0.0 GiB / 32.0 GiB')).toBeTruthy()
  expect(menu.getByRole('region', { name: 'GPU 0 · A' }).querySelector('[style*="width: 0%"]')).toBeTruthy()
  hook.unmount()
})

it('uses first-device compatibility fields only when the array is absent', async () => {
  const legacy = await shown(base)
  expect(legacy.result.current.label).toBe('Σ GPU 10% · VRAM 8.0/32.0 GiB')
  legacy.unmount()
  const empty = await shown({ ...base, gpus: [], uma: true })
  expect(empty.result.current.label).toBe('RAM 64.0/128.0 GiB')
  const menu = render(<>{empty.result.current.menuContent}</>)
  expect(menu.container.textContent).toMatch(/unified|shared/i)
  empty.unmount()
})

it.each([0, 100])('averages %s%% across devices rather than summing utilization', async util => {
  const hook = await shown({
    ...base,
    gpus: [0, 1].map(index => ({
      index,
      name: 'Same GPU',
      util_percent: util,
      vram_used_bytes: 0,
      vram_total_bytes: 32 * GiB
    }))
  })

  expect(hook.result.current.label).toBe(`Σ GPU ${util}% · VRAM 0.0/64.0 GiB`)
  hook.unmount()
})
