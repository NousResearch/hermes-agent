import { describe, expect, it } from 'vitest'

import {
  COMPACT_SCOPE_CONTENT_CLASS,
  COMPACT_SCOPE_TRIGGER_CLASS,
  rosterScopeLabel
} from './scope-label'

const localDefault = {
  connectionId: 'local',
  connectionLabel: 'This device',
  profile: 'default'
}

const remoteBot = {
  connectionId: 'homelab',
  connectionLabel: 'Homelab',
  profile: 'inbox-bot'
}

describe('rosterScopeLabel', () => {
  it('keeps the full non-compact roster labels including (current)', () => {
    expect(rosterScopeLabel(localDefault, 'local', false)).toBe('default — This device (current)')
    expect(rosterScopeLabel(remoteBot, 'local', false)).toBe('inbox-bot — Homelab')
  })

  it('shortens compact labels: no (current), and default profile is just the connection', () => {
    expect(rosterScopeLabel(localDefault, 'local', true)).toBe('This device')
    expect(rosterScopeLabel(remoteBot, 'local', true)).toBe('inbox-bot — Homelab')
    expect(rosterScopeLabel(localDefault, 'local', true)).not.toMatch(/ \(current\)/)
    expect(rosterScopeLabel(localDefault, 'local', true)).not.toMatch(/^default — /)
  })
})

describe('compact scope selector classes', () => {
  it('constrains the plugins trigger inside HALF_COL', () => {
    expect(COMPACT_SCOPE_TRIGGER_CLASS).toMatch(/min-w-0/)
    expect(COMPACT_SCOPE_TRIGGER_CLASS).toMatch(/truncate|overflow-hidden/)
    expect(COMPACT_SCOPE_TRIGGER_CLASS).not.toMatch(/max-w-64/)
  })

  it('caps compact content to the available viewport', () => {
    expect(COMPACT_SCOPE_CONTENT_CLASS).toMatch(
      /max-w-\(--radix-select-available-width\)|max-w-\[min\(var\(--radix-select-available-width\)/
    )
  })
})
