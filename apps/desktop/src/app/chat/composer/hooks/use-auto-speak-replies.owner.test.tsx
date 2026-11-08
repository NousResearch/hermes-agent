import { cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { clearVoiceClientConfigCache } from '@/lib/voice-client-direct'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/store/ambient', () => ({ ownsAmbientCue: async () => true }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

// A Bot chat is owned by (its connection, its profile). The production path —
// the auto-speak hook reading its composer scope, through playSpeechText's
// ladder, down to the REST audio calls — must carry that owner, or the Bot
// speaks with the active profile's voice on the active gateway (#100864).
describe('useAutoSpeakReplies — owner-routed synthesis', () => {
  afterEach(() => {
    cleanup()
    $autoSpeakReplies.set(false)
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    clearVoiceClientConfigCache()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('synthesizes a Bot reply with the scope owner (connection, profile), not the active scope', async () => {
    const api = vi.fn(async ({ path }: { path: string }) =>
      path.startsWith('/api/audio/voice-config') ? { ok: false } : { audio: '' }
    )

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')
    $autoSpeakReplies.set(true)

    const $messages = atom<never[]>([])
    let reply: null | { id: string; pending: boolean; text: string } = null

    const wrapper = ({ children }: { children: ReactNode }) => (
      <ComposerScopeProvider
        value={{ ...MAIN_COMPOSER_SCOPE, $messages, connectionId: 'gw-bots', profile: 'bot-adam', target: 'tile:bot' }}
      >
        {children}
      </ComposerScopeProvider>
    )

    renderHook(
      () =>
        useAutoSpeakReplies({
          conversationActive: false,
          failureLabel: 'failed',
          markSpoken: () => {
            reply = null
          },
          pendingReply: () => reply,
          sessionId: 'bot-session'
        }),
      { wrapper }
    )

    reply = { id: 'm1', pending: false, text: 'Hello from Adam.' }
    $messages.set([])

    await vi.waitFor(() =>
      expect(api.mock.calls.map(([request]) => (request as { path: string }).path)).toContain('/api/audio/speak')
    )

    const scopes = new Set(
      api.mock.calls.map(([request]) => {
        const { connectionId, profile } = request as { connectionId?: string; profile?: string }

        return `${connectionId}::${profile}`
      })
    )

    // voice-config AND the speak POST — every leg names the Bot's owner.
    expect(scopes).toEqual(new Set(['gw-bots::bot-adam']))
  })
})
