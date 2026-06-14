import { QueryClient } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import { notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  setCurrentModel,
  setCurrentProvider
} from '@/store/session'

import { useModelControls } from './use-model-controls'

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn(),
  setGlobalModel: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        modelSwitchFailed: 'Model switch failed'
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

describe('useModelControls.refreshCurrentModel', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  it('applies the global model when there is no active runtime session', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: null,
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
  })

  it('does not clobber the active session footer state with global model info', async () => {
    setCurrentModel('deepseek/deepseek-v4-pro')
    setCurrentProvider('deepseek')
    $activeSessionId.set('runtime-1')
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: 'runtime-1',
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('deepseek/deepseek-v4-pro')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('uses config.set for live session model switches and preserves the explicit provider', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ value: 'deepseek/deepseek-v4-pro' })
    const queryClient = new QueryClient()
    setCurrentModel('qwen/qwen3-coder')
    setCurrentProvider('qwen-oauth')
    $activeSessionId.set('runtime-1')

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: 'runtime-1',
        queryClient,
        requestGateway
      })
    )

    let ok = false
    await act(async () => {
      ok = await result.current.selectModel({
        model: 'deepseek/deepseek-v4-pro',
        persistGlobal: false,
        provider: 'deepseek'
      })
    })

    expect(ok).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      confirm_expensive_model: true,
      key: 'model',
      session_id: 'runtime-1',
      value: 'deepseek/deepseek-v4-pro --provider deepseek'
    })
    expect($currentModel.get()).toBe('deepseek/deepseek-v4-pro')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('rolls back the optimistic live-session switch when config.set does not confirm success', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    const queryClient = new QueryClient()
    setCurrentModel('qwen/qwen3-coder')
    setCurrentProvider('qwen-oauth')
    $activeSessionId.set('runtime-1')

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: 'runtime-1',
        queryClient,
        requestGateway
      })
    )

    let ok = true
    await act(async () => {
      ok = await result.current.selectModel({
        model: 'deepseek/deepseek-v4-pro',
        persistGlobal: false,
        provider: 'deepseek'
      })
    })

    expect(ok).toBe(false)
    expect($currentModel.get()).toBe('qwen/qwen3-coder')
    expect($currentProvider.get()).toBe('qwen-oauth')
    expect(notifyError).toHaveBeenCalled()
  })
})
