import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { MoaConfigResponse, MoaPresetConfig, ModelOptionProvider } from '@/types/hermes'

vi.mock('@/hermes', () => ({
  createMoaModelsSaveRequest: (body: MoaConfigResponse) => () => Promise.resolve({ ...body, ok: true }),
  getMoaModels: vi.fn(),
  saveMoaModels: (body: MoaConfigResponse) => Promise.resolve({ ...body, ok: true }),
  setApiRequestProfile: vi.fn()
}))

import { MoaPresetStudio } from './moa-preset-studio'

const RESERVED_NAMES = ['__proto__', 'constructor', 'toString'] as const

const providers: ModelOptionProvider[] = [
  { authenticated: true, models: ['hermes-4'], name: 'Nous', slug: 'nous' },
  {
    authenticated: true,
    models: ['anthropic/claude-opus-4.8'],
    name: 'OpenRouter',
    slug: 'openrouter'
  }
]

const presetFixture = (enabled = true): MoaPresetConfig => ({
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  enabled,
  fanout: 'user_turn',
  max_tokens: 4096,
  reference_max_tokens: 600,
  reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

const configFixture = (
  presets: Array<readonly [string, MoaPresetConfig]>,
  defaultPreset: string,
  activePreset = defaultPreset
): MoaConfigResponse => ({
  active_preset: activePreset,
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  default_preset: defaultPreset,
  enabled: true,
  max_tokens: 4096,
  presets: Object.fromEntries(presets),
  reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

function renderStudio(config: MoaConfigResponse) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <MoaPresetStudio config={config} providers={providers} />
    </QueryClientProvider>
  )
}

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
})

describe('MoaPresetStudio own-property preset selection', () => {
  it.each(RESERVED_NAMES)('renders an own %s preset as the selected source', name => {
    renderStudio(configFixture([[name, presetFixture()]], name))

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain(name)
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('true')
  })

  it.each(RESERVED_NAMES)('ignores an inherited %s pointer and falls back to the first own preset', name => {
    renderStudio(configFixture([['safe', presetFixture()]], name, name))

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('safe')
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('true')
  })

  it('selects between own prototype-reserved presets without reading an inherited value', async () => {
    renderStudio(
      configFixture(
        [
          ['__proto__', presetFixture()],
          ['constructor', presetFixture(false)],
          ['toString', presetFixture()]
        ],
        '__proto__'
      )
    )

    fireEvent.click(screen.getByRole('combobox', { name: 'Preset' }))
    fireEvent.click(await screen.findByRole('option', { name: 'constructor' }))

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('constructor')
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('false')
    expect((screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement).disabled).toBe(true)
  })
})
