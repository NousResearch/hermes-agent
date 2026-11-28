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

// The single-card chip is byte-identical to upstream: toFixed(0)+'G' helper,
// top-level fields, no 'GiB' anywhere in the string.
const SINGLE_CHIP = 'GPU 10% · 8G/32G'

afterEach(() => {
  vi.restoreAllMocks()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

function mount(hardware: LocalHardware) {
  getLocalHardware.mockResolvedValue(hardware)
  $localModelsEnabled.set(true)
  $statusbarHiddenIds.set([])
  vi.spyOn(globalThis.document, 'hidden', 'get').mockReturnValue(false)
  return renderHook(() => useSystemResourcesStatusbarItem())
}

// Multi-card labels end in 'GiB'; settle on that substring.
async function shownMulti(hardware: LocalHardware) {
  const hook = mount(hardware)
  await waitFor(() => expect(String(hook.result.current.label)).toContain('GiB'))
  return hook
}

// Single-card chips never contain 'GiB'; settle on the exact upstream string.
async function shownSingle(hardware: LocalHardware) {
  const hook = mount(hardware)
  await waitFor(() => expect(hook.result.current.label).toBe(SINGLE_CHIP))
  return hook
}

it('shows both duplicate-name GPUs, physical total, and RAM in the actual menu', async () => {
  const hook = await shownMulti({
    ...base,
    gpus: [
      { index: 0, name: 'Same GPU', util_percent: 10, vram_used_bytes: 8 * GiB, vram_total_bytes: 32 * GiB, vram_free_bytes: 24 * GiB },
      { index: 1, name: 'Same GPU', util_percent: 90, vram_used_bytes: 24 * GiB, vram_total_bytes: 32 * GiB, vram_free_bytes: 8 * GiB }
    ]
  })

  // Aggregate-only chip: mean 50%, physical 32/64 GiB — not a per-index list.
  expect(hook.result.current.label).toBe('Σ GPU 50% · VRAM 32.0/64.0 GiB')
  expect(hook.result.current.title).toContain('arithmetic mean')
  expect(hook.result.current.title).toContain('not pooled inference capacity')
  expect(hook.result.current.menuClassName).toContain('w-80')

  const menu = render(<>{hook.result.current.menuContent}</>)

  // Grouped bordered sections, keyed per index so duplicate names stay distinct.
  expect(menu.getByRole('region', { name: 'GPU 0 · Same GPU' })).toBeTruthy()
  expect(menu.getByRole('region', { name: 'GPU 1 · Same GPU' })).toBeTruthy()
  expect(menu.getByText('Σ All GPUs')).toBeTruthy()

  // RAM row keeps the single-path ' GB ' spelling at every card count.
  expect(menu.getByText('64.0 GB / 128.0 GB')).toBeTruthy()

  for (const [index, util] of [
    [0, 10],
    [1, 90]
  ] as const) {
    const section = within(menu.getByRole('region', { name: `GPU ${index} · Same GPU` }))
    expect(section.getByRole('heading', { name: `GPU ${index} · Same GPU` })).toBeTruthy()
    expect(section.getByText(`${util}%`)).toBeTruthy()
  }

  hook.unmount()
})

it('preserves zero, marks unavailable readings and incomplete sums unknown', async () => {
  const hook = await shownMulti({
    ...base,
    gpus: [
      { index: 0, name: 'A', util_percent: 0, vram_used_bytes: 0, vram_total_bytes: 32 * GiB, vram_free_bytes: 32 * GiB },
      { index: 1, name: 'B', util_percent: null, vram_used_bytes: null, vram_total_bytes: null, vram_free_bytes: null }
    ]
  })

  // GPU 1's missing readings keep every aggregate unknown — zero stays 0,
  // never a partial sum or a silent zero.
  expect(hook.result.current.label).toBe('Σ GPU — · VRAM —/— GiB')

  const menu = render(<>{hook.result.current.menuContent}</>)

  const zero = within(menu.getByRole('region', { name: 'GPU 0 · A' }))
  expect(zero.getByText('0%')).toBeTruthy()
  expect(zero.getByText('0.0 GiB / 32.0 GiB')).toBeTruthy()
  const zeroRegion = menu.getByRole('region', { name: 'GPU 0 · A' })
  // A zero reading is a real number: both of GPU 0's meters (utilization and
  // VRAM) render at 0 width; nothing else in this fixture hits exactly 0%.
  expect(zeroRegion.querySelectorAll('[style*="width: 0%"]')).toHaveLength(2)

  const unknown = within(menu.getByRole('region', { name: 'GPU 1 · B' }))
  expect(unknown.getByText('—')).toBeTruthy()
  expect(unknown.getByText('— / —')).toBeTruthy()

  const aggregate = within(menu.getByRole('region', { name: 'GPU aggregate' }))
  expect(aggregate.getByText('—')).toBeTruthy()
  expect(aggregate.getByText('— / —')).toBeTruthy()

  hook.unmount()
})

it('uses first-device compatibility fields when the array is absent or empty', async () => {
  // Older backends: no gpus field — the top-level fields describe GPU 0, and
  // the chip is the exact upstream single-card string.
  const legacy = await shownSingle(base)
  expect(legacy.result.current.title).toBeUndefined()
  const legacyMenu = render(<>{legacy.result.current.menuContent}</>)
  expect(legacyMenu.queryByRole('region')).toBeNull()
  expect(legacyMenu.getByText('8.0 GB / 32.0 GB')).toBeTruthy()
  legacy.unmount()
  legacyMenu.unmount()

  // An explicit empty array with gpu_name present also falls back to the
  // top-level fields (gpuReadings), and the UMA note renders.
  const empty = await shownSingle({ ...base, gpus: [], uma: true })
  const emptyMenu = render(<>{empty.result.current.menuContent}</>)
  expect(emptyMenu.container.textContent).toMatch(/unified|shared/i)
  expect(emptyMenu.getByText('64.0 GB / 128.0 GB')).toBeTruthy()
  empty.unmount()
})

it.each([
  [
    'new backend, exactly one card',
    (b: LocalHardware): LocalHardware => ({
      ...b,
      gpus: [
        { index: 0, name: 'Same GPU', util_percent: 10, vram_used_bytes: 8 * GiB, vram_total_bytes: 32 * GiB, vram_free_bytes: 24 * GiB }
      ]
    })
  ],
  ['legacy backend, no gpus field', (b: LocalHardware): LocalHardware => b]
])('single-GPU rendering stays byte-identical to upstream (%s)', async (_, make) => {
  const hook = await shownSingle(make(base))

  // The exact upstream chip string — the single-card path is untouched.
  expect(hook.result.current.label).toBe(SINGLE_CHIP)
  expect(hook.result.current.title).toBeUndefined()
  expect(hook.result.current.menuClassName).toBe('w-64 p-0')

  const menu = render(<>{hook.result.current.menuContent}</>)

  // No GPU-indexed sections, no aggregate section, no bordered blocks.
  expect(menu.queryByRole('region')).toBeNull()
  expect(menu.queryByText('Σ All GPUs')).toBeNull()
  expect(menu.queryByText('GPU 0 · Same GPU')).toBeNull()

  // The plain single-card rows: utilization + VRAM + RAM, upstream helpers.
  expect(menu.getByText('10%')).toBeTruthy()
  expect(menu.getByText('8.0 GB / 32.0 GB')).toBeTruthy()
  expect(menu.getByText('64.0 GB / 128.0 GB')).toBeTruthy()

  // The header keeps the single card's name (upstream behavior).
  expect(menu.getByText('Same GPU')).toBeTruthy()

  hook.unmount()
})

it.each([0, 100])('averages %s%% across devices rather than summing utilization', async util => {
  const hook = await shownMulti({
    ...base,
    gpus: [0, 1].map(index => ({
      index,
      name: 'Same GPU',
      util_percent: util,
      vram_used_bytes: 0,
      vram_total_bytes: 32 * GiB,
      vram_free_bytes: 32 * GiB
    }))
  })

  // Mean, not sum: 0+0 stays 0%, 100+100 stays 100%.
  expect(hook.result.current.label).toBe(`Σ GPU ${util}% · VRAM 0.0/64.0 GiB`)
  hook.unmount()
})
