import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { LocalHardware } from '@/types/hermes'

import { LocalModelsHardwareSection } from './local-models-hardware-section'

afterEach(cleanup)

const base: LocalHardware = {
  uma: false,
  vram_total_bytes: 96 << 30,
  vram_usable_bytes: 90 << 30,
  ram_total_bytes: 256 << 30,
  ram_available_bytes: 200 << 30,
  vram_label: '96.0 GB',
  gpu_name: 'NVIDIA RTX PRO 6000 A',
  gpu_util_percent: 0,
  vram_used_bytes: 0
}

it('keeps a one-card machine on a single name', () => {
  render(
    <I18nProvider configClient={null}>
      <LocalModelsHardwareSection hardware={base} />
    </I18nProvider>
  )

  expect(screen.getByText('NVIDIA RTX PRO 6000 A')).toBeTruthy()
  expect(screen.queryByText(/GPU 1/)).toBeNull()
})

it('lists every installed GPU on the machine row', () => {
  render(
    <I18nProvider configClient={null}>
      <LocalModelsHardwareSection
        hardware={{
          ...base,
          gpus: [
            {
              index: 0,
              name: 'NVIDIA RTX PRO 6000 A',
              util_percent: 0,
              vram_used_bytes: 0,
              vram_total_bytes: 96 << 30,
              vram_free_bytes: 96 << 30
            },
            {
              index: 1,
              name: 'NVIDIA RTX PRO 6000 B',
              util_percent: 0,
              vram_used_bytes: 0,
              vram_total_bytes: 96 << 30,
              vram_free_bytes: 96 << 30
            }
          ]
        }}
      />
    </I18nProvider>
  )

  expect(screen.getByText(/GPU 0 · NVIDIA RTX PRO 6000 A/)).toBeTruthy()
  expect(screen.getByText(/GPU 1 · NVIDIA RTX PRO 6000 B/)).toBeTruthy()
})
