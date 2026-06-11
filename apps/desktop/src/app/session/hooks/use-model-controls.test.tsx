import { QueryClient } from '@tanstack/react-query'
import { renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $notifications, clearNotifications } from '@/store/notifications'
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

function wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider initialLocale="en">{children}</I18nProvider>
}

function setup(requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>) {
  const queryClient = new QueryClient()

  return renderHook(
    () =>
      useModelControls({
        activeSessionId: 'sid-1',
        queryClient,
        requestGateway: requestGateway as <T>(method: string, params?: Record<string, unknown>) => Promise<T>
      }),
    { wrapper }
  ).result
}

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
})

describe('useModelControls.selectModel', () => {
  afterEach(() => {
    clearNotifications()
    vi.restoreAllMocks()
  })

  it('rolls back and notifies when slash.exec resolves with a structured error', async () => {
    setCurrentModel('opus-prev')
    setCurrentProvider('anthropic')

    // slash.exec RESOLVES _ok even when the backend rejected the live switch —
    // the failure rides in `result.error`, not a promise rejection.
    const requestGateway = vi.fn().mockResolvedValue({
      output: "  ✗ Model 'bad/model' not available",
      warning: 'live session sync failed: model switch failed',
      error: 'live session sync failed: model switch failed'
    })

    const result = setup(requestGateway)
    const ok = await result.current.selectModel({ model: 'bad/model', provider: 'anthropic', persistGlobal: false })

    expect(ok).toBe(false)
    // The optimistic model is restored — the UI must not sit on a model the
    // backend never selected.
    expect($currentModel.get()).toBe('opus-prev')
    expect($currentProvider.get()).toBe('anthropic')
    // ...and the user sees why.
    const notes = $notifications.get()
    expect(notes[0]?.kind).toBe('error')
    expect(notes[0]?.message).toContain('live session sync failed')
  })

  it('keeps the new model when slash.exec resolves cleanly', async () => {
    setCurrentModel('opus-prev')
    setCurrentProvider('anthropic')

    const requestGateway = vi.fn().mockResolvedValue({ output: '  ✓ Model switched: new/model' })

    const result = setup(requestGateway)
    const ok = await result.current.selectModel({ model: 'new/model', provider: 'openai', persistGlobal: false })

    expect(ok).toBe(true)
    expect($currentModel.get()).toBe('new/model')
    expect($currentProvider.get()).toBe('openai')
    expect($notifications.get()).toHaveLength(0)
  })

  it('does not let slow switch A roll back fast switch B when A fails late', async () => {
    setCurrentModel('opus-prev')
    setCurrentProvider('anthropic')

    // Switch A hangs on a manually-resolved promise; switch B resolves clean
    // immediately. A is then failed AFTER B has fully committed — the exact
    // race from the review: A's late rollback used to restore the PRE-A model,
    // clobbering B's success.
    let resolveA: (value: { error?: string; output?: string }) => void = () => {}
    const slowA = new Promise<{ error?: string; output?: string }>(resolve => {
      resolveA = resolve
    })

    const requestGateway = vi
      .fn()
      .mockReturnValueOnce(slowA)
      .mockResolvedValueOnce({ output: '  ✓ Model switched: model-b' })

    const result = setup(requestGateway)

    const pendingA = result.current.selectModel({ model: 'model-a', provider: 'prov-a', persistGlobal: false })
    const okB = await result.current.selectModel({ model: 'model-b', provider: 'prov-b', persistGlobal: false })

    expect(okB).toBe(true)
    expect($currentModel.get()).toBe('model-b')
    expect($currentProvider.get()).toBe('prov-b')

    resolveA({ error: 'live session sync failed: model switch failed' })
    const okA = await pendingA

    expect(okA).toBe(false)
    // A's late failure must NOT roll back to the pre-A model: B owns the state.
    expect($currentModel.get()).toBe('model-b')
    expect($currentProvider.get()).toBe('prov-b')
    // The user still sees A's failure toast — only the store rollback is skipped.
    const notes = $notifications.get()
    expect(notes).toHaveLength(1)
    expect(notes[0]?.kind).toBe('error')
    expect(notes[0]?.message).toContain('live session sync failed')
  })

  it('ignores a stale late success without disturbing the newer switch', async () => {
    setCurrentModel('opus-prev')
    setCurrentProvider('anthropic')

    let resolveA: (value: { error?: string; output?: string }) => void = () => {}
    const slowA = new Promise<{ error?: string; output?: string }>(resolve => {
      resolveA = resolve
    })

    const requestGateway = vi
      .fn()
      .mockReturnValueOnce(slowA)
      .mockResolvedValueOnce({ output: '  ✓ Model switched: model-b' })

    const result = setup(requestGateway)

    const pendingA = result.current.selectModel({ model: 'model-a', provider: 'prov-a', persistGlobal: false })
    const okB = await result.current.selectModel({ model: 'model-b', provider: 'prov-b', persistGlobal: false })

    expect(okB).toBe(true)

    resolveA({ output: '  ✓ Model switched: model-a' })
    const okA = await pendingA

    // A stale success reports false (callers must not chain follow-up edits
    // onto a selection the UI no longer shows) and leaves B's state intact.
    expect(okA).toBe(false)
    expect($currentModel.get()).toBe('model-b')
    expect($currentProvider.get()).toBe('prov-b')
    expect($notifications.get()).toHaveLength(0)
  })
})
