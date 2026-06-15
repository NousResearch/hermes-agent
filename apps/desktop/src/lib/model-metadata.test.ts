import { describe, expect, it } from 'vitest'

import { modelDescription, routerHostSummary } from './model-metadata'

describe('routerHostSummary', () => {
  it('returns null when a model has no router metadata', () => {
    expect(routerHostSummary({ name: 'AI-Router', slug: 'custom:ai-router', models: ['qwen'] }, 'qwen')).toBeNull()
  })

  it('labels a single router host', () => {
    const summary = routerHostSummary(
      {
        name: 'AI-Router',
        slug: 'custom:ai-router',
        models: ['qwen'],
        model_metadata: {
          qwen: {
            router_backend: 'llama-swap',
            router_host: 'ko-mac',
            router_hosts: ['ko-mac']
          }
        }
      },
      'qwen'
    )

    expect(summary?.label).toBe('ko-mac')
    expect(summary?.title).toContain('AI-Router host: ko-mac')
    expect(summary?.title).toContain('backend: llama-swap')
  })

  it('compacts multiple router hosts', () => {
    const summary = routerHostSummary(
      {
        name: 'AI-Router',
        slug: 'custom:ai-router',
        models: ['qwen'],
        model_metadata: {
          qwen: {
            router_host: 'ko-mac',
            router_hosts: ['ko-mac', 'ko-taro']
          }
        }
      },
      'qwen'
    )

    expect(summary?.label).toBe('ko-mac +1')
    expect(summary?.title).toContain('ko-mac, ko-taro')
  })
})

describe('modelDescription', () => {
  it('returns the trimmed blurb when present', () => {
    expect(
      modelDescription(
        {
          name: 'AI-Router',
          slug: 'custom:ai-router',
          models: ['auto'],
          model_metadata: { auto: { description: '  Picks the best model for each request.  ' } }
        },
        'auto'
      )
    ).toBe('Picks the best model for each request.')
  })

  it('returns null when metadata or the blurb is missing', () => {
    expect(modelDescription({ name: 'AI-Router', slug: 'custom:ai-router', models: ['m'] }, 'm')).toBeNull()
    expect(
      modelDescription(
        {
          name: 'AI-Router',
          slug: 'custom:ai-router',
          models: ['m'],
          model_metadata: { m: { router_host: 'ko-mac' } }
        },
        'm'
      )
    ).toBeNull()
  })

  it('returns null for a whitespace-only blurb', () => {
    expect(
      modelDescription(
        {
          name: 'AI-Router',
          slug: 'custom:ai-router',
          models: ['m'],
          model_metadata: { m: { description: '   ' } }
        },
        'm'
      )
    ).toBeNull()
  })
})
