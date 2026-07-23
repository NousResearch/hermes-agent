import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestProfile } from '@/hermes'
import type { MoaConfigResponse, ModelOptionProvider } from '@/types/hermes'

import { MoaPresetStudio } from './moa-preset-studio'

const providers: ModelOptionProvider[] = [
  { authenticated: true, models: ['hermes-4'], name: 'Nous', slug: 'nous' },
  {
    authenticated: true,
    models: ['anthropic/claude-opus-4.8'],
    name: 'OpenRouter',
    slug: 'openrouter'
  }
]

const configFixture = (): MoaConfigResponse => ({
  active_preset: 'default',
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  default_preset: 'default',
  enabled: true,
  max_tokens: 4096,
  presets: {
    default: {
      aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
      aggregator_temperature: null,
      enabled: true,
      fanout: 'user_turn',
      max_tokens: 4096,
      reference_max_tokens: 600,
      reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
      reference_temperature: null
    }
  },
  reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

function renderStudio() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <MoaPresetStudio config={configFixture()} providers={providers} />
    </QueryClientProvider>
  )
}

describe('MoaPresetStudio real profile-captured save integration', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    api = vi.fn(async (request: { body?: MoaConfigResponse }) => ({ ...request.body, ok: true }))
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestProfile('profile-a')
  })

  afterEach(() => {
    cleanup()
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
  })

  it('saves a profile-A draft to profile A when routing switches immediately after the click', async () => {
    renderStudio()

    fireEvent.click(screen.getByRole('switch', { name: 'Preset enabled' }))
    expect(api).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    setApiRequestProfile('profile-b')

    expect(api).toHaveBeenCalledTimes(1)
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          presets: expect.objectContaining({
            default: expect.objectContaining({ enabled: false })
          })
        }),
        method: 'PUT',
        path: '/api/model/moa',
        profile: 'profile-a'
      })
    )

    await act(async () => Promise.resolve())

    expect(api).toHaveBeenCalledTimes(1)
  })
})
