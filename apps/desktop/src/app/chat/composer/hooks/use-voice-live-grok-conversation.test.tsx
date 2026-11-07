// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { GrokVoiceHandlers } from '@/lib/voice-live-grok'

import { useVoiceLiveGrokConversation } from './use-voice-live-grok-conversation'

const transport = vi.hoisted(() => ({
  handlers: [] as GrokVoiceHandlers[],
  sessions: [] as Array<{
    close: ReturnType<typeof vi.fn>
    setMuted: ReturnType<typeof vi.fn>
    speak: ReturnType<typeof vi.fn>
    useSessionId: ReturnType<typeof vi.fn>
  }>
}))

vi.mock('@/lib/voice-live-grok', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    GrokVoiceSession: class {
      close = vi.fn()
      setMuted = vi.fn()
      speak = vi.fn()
      useSessionId = vi.fn()

      constructor(handlers: GrokVoiceHandlers) {
        transport.handlers.push(handlers)
        transport.sessions.push(this)
      }

      async start(): Promise<void> {}
    }
  }
})

function wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider configClient={null} initialLocale="en">{children}</I18nProvider>
}

interface HookOptions {
  busy?: boolean | (() => boolean)
  onInterrupt?: () => Promise<void>
  onSubmit?: (text: string, context: string) => Promise<void>
  pendingResponse?: () => { id: string; pending: boolean; text: string } | null
  consumePendingResponse?: () => void
}

async function mountSession(options: HookOptions = {}) {
  const hook = renderHook(
    () =>
      useVoiceLiveGrokConversation({
        busy: typeof options.busy === 'function' ? options.busy() : (options.busy ?? false),
        chatSessionId: () => 'chat-1',
        consumePendingResponse: options.consumePendingResponse ?? vi.fn(),
        enabled: false,
        onInterrupt: options.onInterrupt,
        onSubmit: options.onSubmit ?? vi.fn(async () => {}),
        pendingResponse: options.pendingResponse ?? (() => null)
      }),
    { wrapper }
  )

  await act(async () => {
    await hook.result.current.start()
  })

  return {
    ...hook,
    handlers: transport.handlers.at(-1) as GrokVoiceHandlers,
    session: transport.sessions.at(-1)!
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  transport.handlers.length = 0
  transport.sessions.length = 0
  vi.useRealTimers()
})

describe('Grok-Live delegation races', () => {
  it('awaits interruption before clearing the old response and submitting the new turn', async () => {
    let finishInterrupt: (() => void) | undefined

    const onInterrupt = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finishInterrupt = resolve
        })
    )

    const onSubmit = vi.fn(async () => {})
    const consumePendingResponse = vi.fn()

    const { handlers } = await mountSession({
      busy: true,
      consumePendingResponse,
      onInterrupt,
      onSubmit,
      pendingResponse: () => ({ id: 'old-turn', pending: true, text: 'Old partial' })
    })

    act(() => {
      handlers.onDelegation('new-delegation', 'New question', 'User: New question')
    })

    expect(onInterrupt).toHaveBeenCalledOnce()
    expect(consumePendingResponse).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      finishInterrupt?.()
      await Promise.resolve()
    })

    expect(consumePendingResponse).toHaveBeenCalledOnce()
    expect(onSubmit).toHaveBeenCalledWith('New question', 'User: New question')
  })

  it('lets only the newest delegation submit when interruption is still pending', async () => {
    let finishInterrupt: (() => void) | undefined

    const interrupted = new Promise<void>(resolve => {
      finishInterrupt = resolve
    })

    const onSubmit = vi.fn(async () => {})

    const { handlers } = await mountSession({
      busy: true,
      onInterrupt: vi.fn(() => interrupted),
      onSubmit
    })

    act(() => {
      handlers.onDelegation('first', 'First question', 'User: First question')
      handlers.onDelegation('second', 'Second question', 'User: Second question')
    })

    await act(async () => {
      finishInterrupt?.()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(onSubmit).toHaveBeenCalledOnce()
    expect(onSubmit).toHaveBeenCalledWith('Second question', 'User: Second question')
  })

  it('never speaks the interrupted turn final response, then speaks the new response', async () => {
    let busy = true

    let response: { id: string; pending: boolean; text: string } | null = {
      id: 'old-turn',
      pending: true,
      text: 'Old partial'
    }

    const { handlers, rerender, session } = await mountSession({
      busy: () => busy,
      onInterrupt: vi.fn(async () => {}),
      pendingResponse: () => response
    })

    await act(async () => {
      handlers.onDelegation('new-delegation', 'New question', 'User: New question')
      await Promise.resolve()
      await Promise.resolve()
    })

    // Interrupting settles the old assistant row after the new delegation began.
    busy = false
    rerender()
    response = { id: 'old-turn', pending: false, text: 'Stale old answer' }
    await act(async () => {
      vi.advanceTimersByTime(250)
    })
    expect(session.speak).not.toHaveBeenCalled()

    response = { id: 'new-turn', pending: false, text: 'New answer' }
    await act(async () => {
      vi.advanceTimersByTime(250)
    })
    expect(session.speak).toHaveBeenCalledOnce()
    expect(session.speak).toHaveBeenCalledWith('New answer')
  })

  it('suppresses a superseded response whose id first appears during interruption', async () => {
    let response: { id: string; pending: boolean; text: string } | null = null

    const { handlers, rerender, session } = await mountSession({
      busy: true,
      onInterrupt: vi.fn(async () => {
        response = { id: 'late-old', pending: false, text: 'Old interrupted answer' }
      }),
      onSubmit: vi.fn(async () => {}),
      pendingResponse: () => response,
      consumePendingResponse: () => {
        response = null
      }
    })

    act(() => {
      handlers.onDelegation('new', 'New question', 'User: New question')
    })
    rerender({ busy: false })
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      vi.advanceTimersByTime(250)
    })

    expect(session.speak).not.toHaveBeenCalled()
    expect(response).toBeNull()
  })

  it('uses the current mute state for rapid toggles without deferred updater RPCs', async () => {
    const { result, session } = await mountSession()

    act(() => {
      result.current.toggleMute()
      result.current.toggleMute()
    })

    expect(session.setMuted.mock.calls).toEqual([[true], [false]])
    expect(result.current.muted).toBe(false)
  })
})
