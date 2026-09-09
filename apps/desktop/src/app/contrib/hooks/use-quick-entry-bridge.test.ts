import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useQuickEntryBridge, quickEntrySubmitAck } from './use-quick-entry-bridge'
import { sessionTileDelegate } from '@/store/session-states'

const registeredSubmitHandler = vi.hoisted(() => ({
  current: null as ((payload: { correlationId: string; target: string; text: string }) => void) | null
}))

vi.mock('@/store/quick-entry', async importOriginal => {
  const actual = await importOriginal<typeof import('@/store/quick-entry')>()

  return {
    ...actual,
    setQuickEntrySubmitHandler(fn: Parameters<typeof actual.setQuickEntrySubmitHandler>[0]) {
      registeredSubmitHandler.current = fn
      actual.setQuickEntrySubmitHandler(fn)
    }
  }
})

vi.mock('@/store/session-states', () => ({
  sessionTileDelegate: vi.fn()
}))

describe('quickEntrySubmitAck', () => {
  it('reports a rejected prompt as failure instead of acknowledging success', () => {
    expect(quickEntrySubmitAck(false)).toEqual({
      code: 'submit-rejected',
      message: 'The prompt was not accepted.',
      ok: false,
      retryable: true
    })
  })

  it('acknowledges only an accepted prompt as success', () => {
    expect(quickEntrySubmitAck(true)).toEqual({ ok: true })
  })
})

describe('useQuickEntryBridge', () => {
  const originalHermesDesktop = window.hermesDesktop
  let root: Root | null = null

  afterEach(() => {
    root?.unmount()
    root = null
    window.hermesDesktop = originalHermesDesktop
    vi.clearAllMocks()
  })

  async function renderBridge() {
    function Harness() {
      useQuickEntryBridge({
        submitText: () => true,
        submitTextToNewSession: async () => ({
          runtimeSessionId: 'runtime-new',
          sessionId: 'stored-new'
        })
      })

      return null
    }

    const container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    await act(async () => {
      root?.render(createElement(Harness))
    })

    return { container, submit: registeredSubmitHandler.current! }
  }

  it('acknowledges an accepted selected-session submit with exact identity', async () => {
    const correlationId = 'selected-submit-correlation'
    const ackSubmit = vi.fn()
    ackSubmit.mockImplementationOnce(() => {
      throw new Error('ack channel failed')
    })
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-1')
    const submitToSession = vi.fn(async () => 'runtime-session-1')
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    expect(registeredSubmitHandler.current).toBeTypeOf('function')

    await act(async () => {
      await submit({
        correlationId,
        target: 'stored-session-1',
        text: 'Send from Quick Entry'
      })
    })

    expect(resumeTile).toHaveBeenCalledWith('stored-session-1')
    expect(submitToSession).toHaveBeenCalledWith('runtime-session-1', 'Send from Quick Entry')
    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      ok: true,
      runtimeSessionId: 'runtime-session-1',
      sessionId: 'stored-session-1'
    })

    container.remove()
  })

  it('reports the recovered runtime id when the delegate rebinds it', async () => {
    const correlationId = 'selected-submit-recovered'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-before-recovery')
    const submitToSession = vi.fn(async () => 'runtime-session-recovered')
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    await act(async () => {
      await submit({ correlationId, target: 'stored-session-1', text: 'Recover me' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      ok: true,
      runtimeSessionId: 'runtime-session-recovered',
      sessionId: 'stored-session-1'
    })

    container.remove()
  })

  it('keeps a post-dispatch failure non-retryable', async () => {
    const correlationId = 'selected-submit-post-dispatch-failure'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-1')
    const submitToSession = vi.fn(async () => {
      throw new Error('gateway down')
    })
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    await act(async () => {
      await submit({ correlationId, target: 'stored-session-1', text: 'Fail after dispatch' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-failed',
      message: 'The selected session prompt was dispatched, but backend acceptance is unknown.',
      ok: false,
      retryable: false
    })

    container.remove()
  })

  it('keeps a pre-dispatch failure retryable', async () => {
    const correlationId = 'selected-submit-pre-dispatch-failure'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => {
      throw new Error('resume failed')
    })
    const submitToSession = vi.fn(async () => 'runtime-session-1')
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    await act(async () => {
      await submit({ correlationId, target: 'stored-session-1', text: 'Fail before dispatch' })
    })

    expect(submitToSession).not.toHaveBeenCalled()
    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-failed',
      message: 'resume failed',
      ok: false,
      retryable: true
    })

    container.remove()
  })
})
