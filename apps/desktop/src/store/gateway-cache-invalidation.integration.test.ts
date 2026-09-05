import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  beginGatewaySwitch,
  endGatewaySwitch,
  registerGatewaySwitchLifecycle
} from './gateway-switch'
import { clearTranscriptTails, loadTranscriptTail, saveTranscriptTail } from './transcript-tail-cache'

// This is the durable cache consumer at the gateway-switch commit point. The
// fixture never opens a socket or contacts a backend: two fake gateways only
// exercise the same stored-id collision that a source switch can expose.
const message = (id: string): ChatMessage =>
  ({ id, parts: [{ text: id, type: 'text' }], role: 'assistant' }) as unknown as ChatMessage

const oldScope = { connectionId: 'gateway-a', profile: 'default' }
const nextScope = { connectionId: 'gateway-b', profile: 'default' }

beforeEach(() => {
  window.localStorage.clear()
  clearTranscriptTails()
})

afterEach(() => {
  clearTranscriptTails()
})

describe('gateway switch durable-cache invalidation', () => {
  it('clears every old-gateway tail before a replacement can reuse its stored id', () => {
    saveTranscriptTail('same-session', [message('old-a')], oldScope)
    saveTranscriptTail('same-session', [message('old-b')], nextScope)

    expect(loadTranscriptTail('same-session', oldScope)?.[0].id).toBe('old-a')
    expect(loadTranscriptTail('same-session', nextScope)?.[0].id).toBe('old-b')

    const token = beginGatewaySwitch()

    expect(loadTranscriptTail('same-session', oldScope)).toBeNull()
    expect(loadTranscriptTail('same-session', nextScope)).toBeNull()

    // The new source is allowed to paint a fresh tail under the same durable
    // id only after the switch has removed the old source's cache entries.
    saveTranscriptTail('same-session', [message('fresh-b')], nextScope)
    expect(loadTranscriptTail('same-session', nextScope)?.[0].id).toBe('fresh-b')

    endGatewaySwitch(token)
  })

  it('keeps switch lifecycle ordering: outgoing cache is visible to reset, gone before publication', () => {
    saveTranscriptTail('same-session', [message('old')], oldScope)
    const observations: string[] = []
    const off = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: () => {
        observations.push(`before:${loadTranscriptTail('same-session', oldScope)?.[0].id ?? 'missing'}`)
      },
      refreshSessions: async () => undefined
    })

    const token = beginGatewaySwitch()

    observations.push(`after:${loadTranscriptTail('same-session', oldScope)?.[0].id ?? 'missing'}`)
    endGatewaySwitch(token)
    off()

    expect(observations).toEqual(['before:old', 'after:missing'])
  })
})
