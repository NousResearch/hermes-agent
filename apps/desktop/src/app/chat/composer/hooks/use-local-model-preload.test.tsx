import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import { $currentModel, $currentProvider } from '@/store/session'

import { useLocalModelPreload } from './use-local-model-preload'

const preloadLocalModel = vi.hoisted(() => vi.fn())

vi.mock('@/api/local-models', () => ({ preloadLocalModel }))

describe('useLocalModelPreload', () => {
  beforeEach(() => {
    preloadLocalModel.mockReset().mockResolvedValue({ ok: true, status: 'started' })
    $localModelsEnabled.set(true)
    $currentProvider.set('llamacpp')
    $currentModel.set('qwen')
  })

  afterEach(() => {
    cleanup()
    $localModelsEnabled.set(false)
    $currentProvider.set('')
    $currentModel.set('')
  })

  it('asks for the current model once when typing begins, not on every render', () => {
    const { rerender } = renderHook(({ typing }) => useLocalModelPreload(typing), { initialProps: { typing: false } })
    expect(preloadLocalModel).not.toHaveBeenCalled()

    rerender({ typing: true })
    rerender({ typing: true })
    expect(preloadLocalModel).toHaveBeenCalledExactlyOnceWith('qwen')
  })

  it('asks again for a new draft after the composer was emptied', () => {
    const { rerender } = renderHook(({ typing }) => useLocalModelPreload(typing), { initialProps: { typing: true } })
    rerender({ typing: false })
    rerender({ typing: true })
    expect(preloadLocalModel).toHaveBeenCalledTimes(2)
  })

  it('asks for the newly chosen model when it changes mid-draft', () => {
    renderHook(() => useLocalModelPreload(true))
    act(() => $currentModel.set('gemma'))
    expect(preloadLocalModel.mock.calls.map(([id]) => id)).toEqual(['qwen', 'gemma'])
  })

  it.each([
    ['a cloud provider', () => $currentProvider.set('openrouter')],
    ['local models switched off in the app', () => $localModelsEnabled.set(false)],
    ['no model selected', () => $currentModel.set('')]
  ])('does nothing with %s', (_label, arrange) => {
    arrange()
    renderHook(() => useLocalModelPreload(true))
    expect(preloadLocalModel).not.toHaveBeenCalled()
  })

  it('swallows a failed request', async () => {
    preloadLocalModel.mockRejectedValue(new Error('backend down'))
    const unhandled = vi.fn()
    process.on('unhandledRejection', unhandled)

    renderHook(() => useLocalModelPreload(true))
    await new Promise(resolve => setTimeout(resolve, 0))
    process.off('unhandledRejection', unhandled)

    expect(unhandled).not.toHaveBeenCalled()
  })
})
