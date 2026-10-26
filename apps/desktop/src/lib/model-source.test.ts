import { describe, expect, it } from 'vitest'

import { groupModelSources, modelSource } from './model-source'

describe('modelSource', () => {
  it('takes the namespace before the first slash', () => {
    expect(modelSource('wb-proxy/glm-5.3-flash')).toBe('wb-proxy')
    expect(modelSource('opencode-zen-free/mimo-v2.6-flash-free')).toBe('opencode-zen-free')
  })

  it('is null for a bare id, so a flat catalog never splits', () => {
    expect(modelSource('glm-5.3-flash')).toBeNull()
    expect(modelSource('')).toBeNull()
    expect(modelSource('/leading')).toBeNull()
  })
})

describe('groupModelSources', () => {
  it('groups a proxy catalog by upstream, first-appearance order, catalog order inside', () => {
    const groups = groupModelSources([
      'wb-proxy/glm-5.3-flash',
      'wb-proxy/step-5-preview',
      'zcode/GLM-5.3-Flash',
      'opencode-zen-free/glm-5.3-flash-free',
      'qwenwork/qwen3.8-max-preview'
    ])

    expect(groups).toEqual([
      { source: 'wb-proxy', models: ['wb-proxy/glm-5.3-flash', 'wb-proxy/step-5-preview'] },
      { source: 'zcode', models: ['zcode/GLM-5.3-Flash'] },
      { source: 'opencode-zen-free', models: ['opencode-zen-free/glm-5.3-flash-free'] },
      { source: 'qwenwork', models: ['qwenwork/qwen3.8-max-preview'] }
    ])
  })

  it('is null for a single model — a one-row group needs no source label', () => {
    expect(groupModelSources(['wb-proxy/glm-5.3-flash'])).toBeNull()
  })

  it('is null when one id carries no prefix, rather than inventing an unlabeled group', () => {
    expect(groupModelSources(['wb-proxy/glm-5.3-flash', 'glm-5.3'])).toBeNull()
  })

  it('is null when every model shares one upstream — the label would repeat the provider', () => {
    expect(groupModelSources(['wb-proxy/glm-5.3-flash', 'wb-proxy/step-5-preview'])).toBeNull()
  })

  it('is null for an empty catalog', () => {
    expect(groupModelSources([])).toBeNull()
  })
})
