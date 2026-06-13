import { QueryClient } from '@tanstack/react-query'
import { describe, expect, it } from 'vitest'

import type { ModelOptionsResponse } from '@/types/hermes'

function patchModelOptionsCache(
  queryClient: QueryClient,
  key: readonly [string, string],
  provider: string,
  model: string
) {
  const prev = queryClient.getQueryData<ModelOptionsResponse>(key)
  if (!prev?.providers?.length) {
    return
  }

  queryClient.setQueryData<ModelOptionsResponse>(key, { ...prev, provider, model })
}

describe('useModelControls cache updates', () => {
  it('does not write a provider-less stub into the model-options cache', () => {
    const queryClient = new QueryClient()

    patchModelOptionsCache(queryClient, ['model-options', 'global'], 'openai', 'gpt-4.1')

    expect(queryClient.getQueryData(['model-options', 'global'])).toBeUndefined()
  })

  it('preserves providers when patching the active model', () => {
    const queryClient = new QueryClient()
    const seeded: ModelOptionsResponse = {
      model: 'old-model',
      provider: 'openai',
      providers: [{ models: ['old-model', 'new-model'], name: 'OpenAI', slug: 'openai' }]
    }

    queryClient.setQueryData(['model-options', 'global'], seeded)
    patchModelOptionsCache(queryClient, ['model-options', 'global'], 'openai', 'new-model')

    expect(queryClient.getQueryData<ModelOptionsResponse>(['model-options', 'global'])).toEqual({
      ...seeded,
      model: 'new-model',
      provider: 'openai'
    })
  })
})
