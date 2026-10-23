import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { BotRow } from '@/plugins/hermes-bots/bot-row'
import { translateBotsIn } from '@/plugins/hermes-bots/i18n-test-helper'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { host } from '@/sdk'
import { $selectedStoredSessionId, setActiveSessionId, setSessionOwnerHint } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

vi.mock('@hermes/plugin-sdk', async importOriginal => ({
  ...(await importOriginal<typeof HermesSdk>()),
  usePluginI18n: () => translateBotsIn('en')
}))

const noop = () => undefined

describe('canonical Bot Chat presence across focus changes', () => {
  it('keeps a working background face while an idle bot stays idle, then settles on completion', () => {
    clearAllSessionStates()

    const working: RosterRow = {
      name: 'researcher',
      connectionId: 'local',
      canonical_session: { id: 'chat-root', resolved_id: 'chat-tip', last_active: 1 }
    }

    const idle: RosterRow = {
      name: 'writer',
      connectionId: 'local',
      canonical_session: { id: 'idle-chat', last_active: 1 }
    }

    setSessionOwnerHint('chat-tip', { connectionId: 'local', profile: 'researcher' })
    setSessionOwnerHint('idle-chat', { connectionId: 'local', profile: 'writer' })
    setActiveSessionId('rt-working')
    $selectedStoredSessionId.set('chat-tip')
    publishSessionState('rt-working', { ...createClientSessionState('chat-tip'), busy: true })
    publishSessionState('rt-idle', createClientSessionState('idle-chat'))

    const { container } = render(
      <>
        {[working, idle].map(bot => (
          <BotRow bot={bot} key={bot.name} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
        ))}
      </>
    )

    const moods = () => [...container.querySelectorAll('[data-hb-mood]')].map(el => el.getAttribute('data-hb-mood'))

    try {
      expect(moods()).toEqual(['think', 'idle'])
      act(() => {
        setActiveSessionId('rt-idle')
        $selectedStoredSessionId.set('idle-chat')
      })
      expect(host.state.busy.get()).toBe(false)
      expect(moods()).toEqual(['think', 'idle'])
      act(() => publishSessionState('rt-working', createClientSessionState('chat-tip')))
      expect(moods()).toEqual(['idle', 'idle'])
    } finally {
      act(() => {
        setActiveSessionId(null)
        $selectedStoredSessionId.set(null)
        clearAllSessionStates()
      })
    }
  })
})
