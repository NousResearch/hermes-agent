import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, patchOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState } from '../app/uiStore.js'

const ref = <T>(current: T) => ({ current })

const handlerCtx = (
  request: (method: string, params?: Record<string, unknown>) => Promise<unknown>,
  submit: () => void
) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { isCanonical: true, request },
      rpc: (method: string, params?: Record<string, unknown>) => request(method, params)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: submit } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const limitEvent = {
  payload: {
    error_surface: {
      code: 'usage_window',
      usage_limit: {
        gauge_id: 'day',
        limit_name: 'daily',
        message: 'You hit the daily limit.',
        scope: 'all',
        suggestions: [{ label: 'Free model', model: 'free/model', note: '', provider: 'nous' }]
      }
    },
    text: 'You hit the daily limit.'
  },
  session_id: 'sess-1',
  type: 'message.complete'
}

describe('the usage-limit card on a canonical connection', () => {
  afterEach(() => {
    patchOverlayState({ confirm: null })
  })

  it('switches this session with session.mutate (the owner refuses config.set model) and re-sends after it applied', async () => {
    patchUiState({ info: { model: 'paid/model' } as any, sid: 'sess-1' })
    const submit = vi.fn()

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { execution_generation: 2, revision: 4 }
      }

      if (method === 'session.mutate') {
        return { execution_generation: 2, model: (params?.payload as { model: string }).model, status: 'applied' }
      }

      throw Object.assign(new Error('use_session_mutation_model'), { data: { reason: 'use_session_mutation_model' } })
    })

    const ctx = handlerCtx(request, submit)

    createGatewayEventHandler(ctx)(limitEvent as any)
    getOverlayState().confirm!.onConfirm()

    await vi.waitFor(() => expect(submit).toHaveBeenCalledWith('/retry'))
    expect(request).not.toHaveBeenCalledWith('config.set', expect.anything())
    expect(request).toHaveBeenCalledWith(
      'session.mutate',
      expect.objectContaining({
        operation: 'model',
        payload: { model: 'free/model', provider: 'nous' },
        session_id: 'sess-1'
      })
    )
    expect(getUiState().info?.model).toBe('free/model')
  })
  it('applies nothing to, and never retries, a chat the user opened while the switch was pending', async () => {
    patchUiState({ info: { model: 'paid/model' } as any, sid: 'sess-1' })
    const submit = vi.fn()
    let commit!: () => void

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { execution_generation: 2, revision: 4 }
      }

      await new Promise<void>(resolve => (commit = resolve))

      return { execution_generation: 2, model: (params?.payload as { model: string }).model, status: 'applied' }
    })

    createGatewayEventHandler(handlerCtx(request, submit))(limitEvent as any)
    getOverlayState().confirm!.onConfirm()
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.mutate', expect.anything()))

    patchUiState({ info: { model: 'other/model' } as any, sid: 'sess-2' })
    commit()
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(submit).not.toHaveBeenCalled()
    expect(getUiState().info?.model).toBe('other/model')
  })
})
