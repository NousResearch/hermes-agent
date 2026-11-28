import { afterEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $petOwnerKey, petProfile } from './pet'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from './session'

function chat(id: string, profile: string): SessionInfo {
  return {
    id,
    profile,
    connection_id: 'local',
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 0,
    title: 'Bot Chat',
    tool_call_count: 0
  }
}

const previous = {
  profile: $activeGatewayProfile.get(),
  active: $activeSessionId.get(),
  selected: $selectedStoredSessionId.get(),
  sessions: $sessions.get()
}

afterEach(() => {
  $activeSessionId.set(previous.active)
  $selectedStoredSessionId.set(previous.selected)
  $sessions.set(previous.sessions)
  $activeGatewayProfile.set(previous.profile)
})

describe('pet profile follows the visible conversation', () => {
  it('does not borrow the launch profile pet for a bot chat', () => {
    $activeGatewayProfile.set('solara')
    $sessions.set([chat('gm-chat', 'grand-master')])
    $selectedStoredSessionId.set('gm-chat')
    $activeSessionId.set(null)
    expect(petProfile()).toBe('grand-master')
  })

  it('changes pets when bot chats switch without changing the gateway', () => {
    $activeGatewayProfile.set('solara')
    $sessions.set([chat('gm-chat', 'grand-master'), chat('aura-chat', 'aura')])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set('gm-chat')
    expect(petProfile()).toBe('grand-master')
    const key = $petOwnerKey.get()
    $selectedStoredSessionId.set('aura-chat')
    expect(petProfile()).toBe('aura')
    expect($petOwnerKey.get()).not.toBe(key)
    expect($activeGatewayProfile.get()).toBe('solara')
  })

  it('uses the gateway profile for a fresh draft without a conversation owner', () => {
    $activeGatewayProfile.set('solara')
    $sessions.set([])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    expect(petProfile()).toBe('solara')
  })
})
