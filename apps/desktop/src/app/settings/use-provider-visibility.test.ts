// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { $visibleModels } from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'

import { useProviderModelVisibility } from './use-provider-visibility'

const providers: ModelOptionProvider[] = [
  { slug: 'openai', name: 'OpenAI', models: ['gpt-4', 'gpt-3.5'] },
  { slug: 'nous', name: 'Nous', models: ['hermes-3'] }
]

describe('useProviderModelVisibility', () => {
  beforeEach(() => {
    localStorage.clear()
    $visibleModels.set(null)
  })

  it('reports visibility from effectiveVisibleKeys (defaults when uncustomized)', () => {
    const { result } = renderHook(() => useProviderModelVisibility('openai', providers))

    expect(result.current.isVisible('gpt-4')).toBe(true)
    expect(result.current.isVisible('gpt-3.5')).toBe(true)
    expect(result.current.visibleCount).toBe(2)
    expect(result.current.allHidden).toBe(false)
  })

  it('toggling the last model sets the allHidden sentinel', () => {
    const { result } = renderHook(() => useProviderModelVisibility('openai', providers))

    act(() => result.current.toggle('gpt-4'))
    act(() => result.current.toggle('gpt-3.5'))

    expect(result.current.isVisible('gpt-4')).toBe(false)
    expect(result.current.isVisible('gpt-3.5')).toBe(false)
    expect(result.current.visibleCount).toBe(0)
    expect(result.current.allHidden).toBe(true)
  })

  it('re-enabling a model clears the sentinel and restores only that model', () => {
    const { result } = renderHook(() => useProviderModelVisibility('openai', providers))

    act(() => {
      result.current.toggle('gpt-4')
      result.current.toggle('gpt-3.5')
    })
    act(() => result.current.toggle('gpt-4'))

    expect(result.current.isVisible('gpt-4')).toBe(true)
    expect(result.current.isVisible('gpt-3.5')).toBe(false)
    expect(result.current.allHidden).toBe(false)
  })

  it('is a no-op and reports hidden when providerSlug is null', () => {
    const { result } = renderHook(() => useProviderModelVisibility(null, providers))

    expect(result.current.isVisible('gpt-4')).toBe(false)
    expect(result.current.visibleCount).toBe(0)

    act(() => result.current.toggle('gpt-4'))

    expect(result.current.isVisible('gpt-4')).toBe(false)
  })

  it('reports all hidden and ignores toggles when the provider is disabled', () => {
    const { result } = renderHook(() => useProviderModelVisibility('openai', providers, false))

    expect(result.current.isVisible('gpt-4')).toBe(false)
    expect(result.current.isVisible('gpt-3.5')).toBe(false)
    expect(result.current.visibleCount).toBe(0)
    expect(result.current.allHidden).toBe(true)

    act(() => result.current.toggle('gpt-4'))

    // Still hidden — toggling a disabled provider must not mutate $visibleModels.
    expect(result.current.isVisible('gpt-4')).toBe(false)
    expect(result.current.visibleCount).toBe(0)
  })
})
