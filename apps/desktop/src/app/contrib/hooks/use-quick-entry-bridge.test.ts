import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useQuickEntryBridge, quickEntrySubmitAck } from './use-quick-entry-bridge'
import { sessionTileDelegate } from '@/store/session-states'

const registeredSubmitHandler = vi.hoisted(() => ({
  current: null as ((payload: { correlationId: string; target: string; text: string }) => void) | null
}))

type SubmitTextForBridge = Parameters<typeof useQuickEntryBridge>[0]['submitText']
type SubmitTextToNewSessionForBridge = Parameters<typeof useQuickEntryBridge>[0]['submitTextToNewSession']

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

  it('names the accepted identity on an accepted ack', () => {
    expect(quickEntrySubmitAck(true, { runtimeSessionId: 'rt-1', storedSessionId: 'st-1' })).toEqual({
      ok: true,
      runtimeSessionId: 'rt-1',
      sessionId: 'st-1'
    })
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

  async function renderBridge(
    submitText: SubmitTextForBridge = () => true,
    submitTextToNewSession: SubmitTextToNewSessionForBridge = async () => ({
      runtimeSessionId: 'runtime-new',
      sessionId: 'stored-new'
    })
  ) {
    function Harness() {
      useQuickEntryBridge({
        submitText,
        submitTextToNewSession
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

  it('acknowledges a current-chat submit with the identity the pipeline accepted', async () => {
    const correlationId = 'current-submit-correlation'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const submitText = vi.fn(async (text: string, options?: Parameters<SubmitTextForBridge>[1]) => {
      options?.onAccepted?.({ runtimeSessionId: 'rt-current-1', storedSessionId: 'st-current-1' })
      expect(text).toBe('Send to current chat')
      return true
    })
    const { container, submit } = await renderBridge(submitText)

    await act(async () => {
      await submit({ correlationId, target: 'current', text: 'Send to current chat' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      ok: true,
      runtimeSessionId: 'rt-current-1',
      sessionId: 'st-current-1'
    })

    container.remove()
  })

  it('passes the correlation id as the new-session pin owner', async () => {
    const correlationId = 'new-submit-correlation'
    const ackSubmit = vi.fn()
    const submitTextToNewSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-new',
      sessionId: 'stored-new'
    }))
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const { container, submit } = await renderBridge(vi.fn(async () => true), submitTextToNewSession)

    await act(async () => {
      await submit({ correlationId, target: 'new', text: 'Send to a new chat' })
    })

    expect(submitTextToNewSession).toHaveBeenCalledWith('Send to a new chat', correlationId)
    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      ok: true,
      runtimeSessionId: 'runtime-new',
      sessionId: 'stored-new'
    })

    container.remove()
  })

  it('acknowledges a current-chat submit with no identity without inventing one', async () => {
    const correlationId = 'current-submit-no-identity'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const { container, submit } = await renderBridge(vi.fn(async () => true))

    await act(async () => {
      await submit({ correlationId, target: 'current', text: 'No identity path' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, { ok: true })

    container.remove()
  })

  it('keeps a rejected current-chat submit retryable', async () => {
    const correlationId = 'current-submit-rejected'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const { container, submit } = await renderBridge(vi.fn(async () => false))

    await act(async () => {
      await submit({ correlationId, target: 'current', text: 'Rejected prompt' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-rejected',
      message: 'The prompt was not accepted.',
      ok: false,
      retryable: true
    })

    container.remove()
  })

  it('reports a failed current-chat submit as retryable', async () => {
    const correlationId = 'current-submit-failed'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const { container, submit } = await renderBridge(
      vi.fn(async () => {
        throw new Error('gateway down')
      })
    )

    await act(async () => {
      await submit({ correlationId, target: 'current', text: 'Failed prompt' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-failed',
      message: 'gateway down',
      ok: false,
      retryable: true
    })

    container.remove()
  })

  it('acknowledges with the proven identity when the accepted stored id matches', async () => {
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
    const submitToSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-session-1',
      storedSessionId: 'stored-session-1'
    }))
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
    const submitToSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-session-recovered',
      storedSessionId: 'stored-session-1'
    }))
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

  it('refuses to acknowledge success when the accepted stored id differs from the requested target', async () => {
    const correlationId = 'selected-submit-identity-mismatch'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-1')
    const submitToSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-session-1',
      storedSessionId: 'other-session'
    }))
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    await act(async () => {
      await submit({ correlationId, target: 'stored-session-1', text: 'Wrong target' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-identity-mismatch',
      message: 'The prompt was accepted by a different session than the one requested.',
      ok: false,
      retryable: false
    })

    container.remove()
  })

  it('refuses to acknowledge success when the accepted stored id is unknown', async () => {
    const correlationId = 'selected-submit-unknown-identity'
    const ackSubmit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        ackSubmit,
        pushState: vi.fn()
      }
    } as unknown as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-1')
    const submitToSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-session-1',
      storedSessionId: null
    }))
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as unknown as ReturnType<typeof sessionTileDelegate>)

    const { container, submit } = await renderBridge()

    await act(async () => {
      await submit({ correlationId, target: 'stored-session-1', text: 'Unknown target' })
    })

    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-identity-mismatch',
      message: 'The prompt was accepted by a different session than the one requested.',
      ok: false,
      retryable: false
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
    const submitToSession = vi.fn(async () => ({
      runtimeSessionId: 'runtime-session-1',
      storedSessionId: 'stored-session-1'
    }))
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
