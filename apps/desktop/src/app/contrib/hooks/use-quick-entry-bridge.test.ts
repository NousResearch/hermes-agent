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

  it('does not acknowledge a selected-session void submit as success', async () => {
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
    } as typeof window.hermesDesktop

    const resumeTile = vi.fn(async () => 'runtime-session-1')
    const submitToSession = vi.fn(async () => {})
    vi.mocked(sessionTileDelegate).mockReturnValue({
      resumeTile,
      submitToSession
    } as ReturnType<typeof sessionTileDelegate>)

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

    expect(registeredSubmitHandler.current).toBeTypeOf('function')

    await act(async () => {
      await registeredSubmitHandler.current?.({
        correlationId,
        target: 'stored-session-1',
        text: 'Send from Quick Entry'
      })
    })

    expect(resumeTile).toHaveBeenCalledWith('stored-session-1')
    expect(submitToSession).toHaveBeenCalledWith('runtime-session-1', 'Send from Quick Entry')
    expect(ackSubmit).toHaveBeenCalledTimes(1)
    expect(ackSubmit).toHaveBeenCalledWith(correlationId, {
      code: 'submit-failed',
      message: 'The selected session prompt was dispatched, but backend acceptance is unknown.',
      ok: false,
      retryable: false
    })

    container.remove()
  })
})
