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

const constructed = vi.hoisted(() => ({ owners: [] as Array<null | { connectionId?: null | string; profile?: null | string }> }))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()

      constructor(_handlers: VoiceLiveHandlers, owner: null | { connectionId?: null | string; profile?: null | string } = null) {
        constructed.owners.push(owner)
      }

      async start(): Promise<void> {}
    }
  }
})

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  constructed.owners.length = 0
})

function mountLive(wrapper?: ({ children }: { children: ReactNode }) => ReactNode) {
  return renderHook(
    () =>
      useVoiceLiveConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit: vi.fn(),
        pendingResponse: () => null,
        seedHistory: () => []
      }),
    wrapper ? { wrapper } : undefined
  )
}

describe('useVoiceLiveConversation — owner-routed session', () => {
  it('constructs the session with the composer scope owner (connection, profile)', async () => {
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ComposerScopeProvider
        value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'gw-bots', profile: 'bot-adam', target: 'tile:bot' }}
      >
        {children}
      </ComposerScopeProvider>
    )

    const hook = mountLive(wrapper)

    await act(async () => {
      await hook.result.current.start()
    })

    expect(constructed.owners).toEqual([{ connectionId: 'gw-bots', profile: 'bot-adam' }])
  })
})
