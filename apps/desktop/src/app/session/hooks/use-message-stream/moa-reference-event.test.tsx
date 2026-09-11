import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

describe('useMessageStream moa.reference accumulation (#64658)', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('keeps every reference model labelled block instead of only the latest one', () => {
    mountStream()

    emit('moa.reference', { count: 2, index: 1, label: 'model-a', text: 'advice-a' })
    emit('moa.reference', { count: 2, index: 2, label: 'model-b', text: 'advice-b' })

    const text = stream.reasoningText()

    expect(text).toContain('model-a')
    expect(text).toContain('advice-a')
    expect(text).toContain('model-b')
    expect(text).toContain('advice-b')
  })

  it('handles a single-reference MoA turn (count=1) without regression', () => {
    mountStream()

    emit('moa.reference', { count: 1, index: 1, label: 'model-a', text: 'only-advice' })

    const text = stream.reasoningText()

    expect(text).toContain('model-a')
    expect(text).toContain('only-advice')
  })

  it('accumulates three or more references in order', () => {
    mountStream()

    emit('moa.reference', { count: 3, index: 1, label: 'model-a', text: 'advice-a' })
    emit('moa.reference', { count: 3, index: 2, label: 'model-b', text: 'advice-b' })
    emit('moa.reference', { count: 3, index: 3, label: 'model-c', text: 'advice-c' })

    const text = stream.reasoningText()

    const orderOk =
      text.indexOf('advice-a') < text.indexOf('advice-b') && text.indexOf('advice-b') < text.indexOf('advice-c')

    expect(text).toContain('advice-a')
    expect(text).toContain('advice-b')
    expect(text).toContain('advice-c')
    expect(orderOk).toBe(true)
  })
})
