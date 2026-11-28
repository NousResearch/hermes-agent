import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import { $preloadingModel } from '@/store/local-model-preload'
import { $currentModel, $currentProvider } from '@/store/session'

import { ModelPreloadHint } from '../model-preload-hint'

import { useLocalModelPreload } from './use-local-model-preload'

const preloadLocalModel = vi.hoisted(() => vi.fn())
const getLocalModelsStatus = vi.hoisted(() => vi.fn())

vi.mock('@/api/local-models', () => ({ getLocalModelsStatus, preloadLocalModel }))

const LOADING = { stage: 'loading', value: 0.4, percent: 40 }

describe('useLocalModelPreload', () => {
  beforeEach(() => {
    preloadLocalModel.mockReset().mockResolvedValue({ ok: true, status: 'started' })
    getLocalModelsStatus.mockReset().mockResolvedValue({ loading: {} })
    $localModelsEnabled.set(true)
    $currentProvider.set('llamacpp')
    $currentModel.set('qwen')
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $localModelsEnabled.set(false)
    $currentProvider.set('')
    $currentModel.set('')
    $preloadingModel.set(null)
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
    expect($preloadingModel.get()).toBeNull()
  })

  describe('the loading hint', () => {
    beforeEach(() => vi.useFakeTimers())

    it('names the model while the server reports it loading, then clears once it stops', async () => {
      getLocalModelsStatus
        .mockResolvedValueOnce({ loading: { qwen: LOADING } })
        .mockResolvedValueOnce({ loading: { qwen: LOADING } })
        .mockResolvedValue({ loading: {} })

      renderHook(() => useLocalModelPreload(true))
      await vi.advanceTimersByTimeAsync(0)
      expect($preloadingModel.get()).toBe('qwen')

      await vi.advanceTimersByTimeAsync(1_000)
      expect($preloadingModel.get()).toBe('qwen')

      await vi.advanceTimersByTimeAsync(1_000)
      expect($preloadingModel.get()).toBeNull()

      // Three polls that miss the model end the watch: no polling goes on forever.
      await vi.advanceTimersByTimeAsync(10_000)
      expect(getLocalModelsStatus).toHaveBeenCalledTimes(5)
    })

    it('keeps watching through status calls that time out on a busy backend', async () => {
      getLocalModelsStatus
        .mockRejectedValueOnce(new Error('Timed out connecting to Hermes backend after 5000ms'))
        .mockRejectedValueOnce(new Error('Timed out connecting to Hermes backend after 5000ms'))
        .mockRejectedValueOnce(new Error('Timed out connecting to Hermes backend after 5000ms'))
        .mockRejectedValueOnce(new Error('Timed out connecting to Hermes backend after 5000ms'))
        .mockResolvedValue({ loaded_models: { qwen: 'loading' }, loading: {} })

      renderHook(() => useLocalModelPreload(true))
      await vi.advanceTimersByTimeAsync(4_000)

      expect($preloadingModel.get()).toBe('qwen')
    })

    it('also trusts the router status when the progress stream has not reported the load yet', async () => {
      getLocalModelsStatus.mockResolvedValue({ loaded_models: { qwen: 'loading' }, loading: {} })

      renderHook(() => useLocalModelPreload(true))
      await vi.advanceTimersByTimeAsync(0)

      expect($preloadingModel.get()).toBe('qwen')
    })

    it('does not show the hint for a model the router already holds loaded', async () => {
      preloadLocalModel.mockResolvedValue({ ok: true, status: 'held' })
      getLocalModelsStatus.mockResolvedValue({ loaded_models: { qwen: 'loaded' }, loading: {} })

      renderHook(() => useLocalModelPreload(true))
      await vi.advanceTimersByTimeAsync(0)

      expect($preloadingModel.get()).toBeNull()
    })

    it('clears the hint and stops polling when typing stops', async () => {
      getLocalModelsStatus.mockResolvedValue({ loading: { qwen: LOADING } })

      const { rerender } = renderHook(({ typing }) => useLocalModelPreload(typing), { initialProps: { typing: true } })
      await vi.advanceTimersByTimeAsync(0)
      expect($preloadingModel.get()).toBe('qwen')

      rerender({ typing: false })
      expect($preloadingModel.get()).toBeNull()

      const polls = getLocalModelsStatus.mock.calls.length
      await vi.advanceTimersByTimeAsync(5_000)
      expect(getLocalModelsStatus).toHaveBeenCalledTimes(polls)
    })

    it.each(['at_cap', 'disabled', 'no_supervisor'])('does not watch when the backend answers %s', async status => {
      preloadLocalModel.mockResolvedValue({ ok: true, status })

      renderHook(() => useLocalModelPreload(true))
      await vi.advanceTimersByTimeAsync(0)

      expect(getLocalModelsStatus).not.toHaveBeenCalled()
      expect($preloadingModel.get()).toBeNull()
    })

    it('shows the hint above the composer only while a model is preloading', () => {
      render(<ModelPreloadHint />)
      expect(screen.queryByRole('status')).toBeNull()

      act(() => $preloadingModel.set('qwen'))
      expect(screen.getByRole('status').textContent).toContain('qwen')

      act(() => $preloadingModel.set(null))
      expect(screen.queryByRole('status')).toBeNull()
    })
  })
})
