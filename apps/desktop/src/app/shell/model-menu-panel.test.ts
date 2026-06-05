import { describe, expect, it } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { groupModels } from './model-menu-panel'

describe('model-menu-panel', () => {
  it('keeps the current provider first in the dropdown', () => {
    const providers: ModelOptionProvider[] = [
      { slug: 'anthropic', name: 'Anthropic', models: ['anthropic/claude-sonnet'] },
      { slug: 'nvidia', name: 'NVIDIA NIM', models: ['nvidia/nemotron-3-ultra-550b-a55b'] },
      { slug: 'google', name: 'Google', models: ['google/gemini-3-pro'] }
    ]

    const groups = groupModels(
      providers,
      '',
      { provider: 'nvidia', model: 'nvidia/nemotron-3-ultra-550b-a55b' },
      null
    )

    expect(groups.map(group => group.provider.slug)).toEqual(['nvidia', 'anthropic', 'google'])
  })
})
