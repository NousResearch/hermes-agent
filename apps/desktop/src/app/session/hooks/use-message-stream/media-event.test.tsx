import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { mediaCardMeta, resetMediaDeliverables } from '@/lib/media-store'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'media-session'

describe('media.deliverable events (M4)', () => {
  let stream: MessageStreamHarness

  beforeEach(async () => {
    resetMediaDeliverables()
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    resetMediaDeliverables()
  })

  const event = (payload: Record<string, unknown>, sessionId?: string) =>
    act(() =>
      stream.handleEvent({
        payload: { timestamp: 100, ...payload },
        session_id: sessionId ?? SID,
        type: 'media.deliverable'
      })
    )

  it('records deliverable metadata into the registry', () => {
    event({ kind: 'image', mime: 'image/png', path: '/tmp/hermes-media/a.png', size: 1234 })

    expect(mediaCardMeta('/tmp/hermes-media/a.png')).toMatchObject({
      kind: 'image',
      path: '/tmp/hermes-media/a.png',
      size: 1234
    })
  })

  it('does not record events with no valid path', () => {
    event({ kind: 'image', path: '' })

    expect(mediaCardMeta('')).toBeNull()
  })

  it('unscoped media events are routed, not dropped (unlike subagent.*)', () => {
    event({ path: '/tmp/hermes-media/b.png' }, '')

    expect(mediaCardMeta('/tmp/hermes-media/b.png')).not.toBeNull()
  })
})
