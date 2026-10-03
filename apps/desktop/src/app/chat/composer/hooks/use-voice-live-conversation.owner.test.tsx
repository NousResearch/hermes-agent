// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { VoiceLiveHandlers } from '@/lib/voice-live'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useVoiceLiveConversation } from './use-voice-live-conversation'

// A Bot chat's GPT-Live session must dial the Bot's own (connection, profile)
// — the same owner route the TTS legs already trust (#117014) — so the voice
// configured on the Bot's profile is the voice that answers (#117401).

const constructed = vi.hoisted(() => ({
  sessions: [] as Array<{
    close: ReturnType<typeof vi.fn>
    handlers: VoiceLiveHandlers
    owner: null | { connectionId?: null | string; profile?: null | string }
  }>
}))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()
      handlers: VoiceLiveHandlers
      owner: null | { connectionId?: null | string; profile?: null | string }

      constructor(
        handlers: VoiceLiveHandlers,
        owner: null | { connectionId?: null | string; profile?: null | string } = null
      ) {
        this.handlers = handlers
        this.owner = owner
        constructed.sessions.push(this)
      }

      async start(): Promise<void> {}
    }
  }
})

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  constructed.sessions.length = 0
})

describe('useVoiceLiveConversation — owner-routed session', () => {
  it('keeps an A conversation and its delegated submit on A across an A→B→A scope switch', async () => {
    const ownerA = { connectionId: 'gateway-a', profile: 'shared', target: 'tile:same-session' as const }
    const ownerB = { connectionId: 'gateway-b', profile: 'shared', target: 'tile:same-session' as const }
    let owner = ownerA
    const submitA = vi.fn()
    const submitB = vi.fn()
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, ...owner }}>{children}</ComposerScopeProvider>
    )

    const hook = renderHook(
      () =>
        useVoiceLiveConversation({
          busy: false,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: owner.connectionId === ownerA.connectionId ? submitA : submitB,
          pendingResponse: () => null,
          seedHistory: () => []
        }),
      { wrapper }
    )

    await act(async () => {
      await hook.result.current.start()
    })

    expect(constructed.sessions.map(session => session.owner)).toEqual([
      { connectionId: 'gateway-a', profile: 'shared' }
    ])

    owner = ownerB
    hook.rerender()

    act(() => {
      constructed.sessions[0].handlers.onDelegation('same-raw-session', [
        { endMs: 1, speaker: 'user', startMs: 0, text: 'route this to A' }
      ])
    })

    expect(submitA).toHaveBeenCalledWith('route this to A', 'User: route this to A')
    expect(submitB).not.toHaveBeenCalled()

    owner = ownerA
    hook.rerender()
    await act(async () => hook.result.current.end())
    expect(constructed.sessions[0].close).toHaveBeenCalledOnce()
  })
})
