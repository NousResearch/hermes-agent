import type { GatewayEventMap, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { moaPhasePayload, moaProgressPayload, moaReferencePayload } from '@/test/contract'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit<K extends GatewayEventName>(type: K, payload: GatewayEventMap[K]) {
  act(() => stream.emit(type, payload, SID))
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useMessageStream moa.progress / moa.phase surfacing', () => {
  it('shows refs k/n lines in the reasoning block as references complete', () => {
    mountStream()

    emit('message.start', {})
    emit('moa.progress', moaProgressPayload({ label: 'model-a', refs_done: 1, refs_total: 3 }))
    emit('moa.progress', moaProgressPayload({ label: 'model-b', refs_done: 2, refs_total: 3 }))

    const text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/3 — model-a')
    expect(text).toContain('MoA refs 2/3 — model-b')
  })

  it('restarts the progress block on the first ref of a new fan-out', () => {
    mountStream()

    emit('message.start', {})
    emit('moa.progress', moaProgressPayload({ label: 'stale', refs_done: 1, refs_total: 2 }))
    emit('moa.progress', moaProgressPayload({ label: 'stale-2', refs_done: 2, refs_total: 2 }))
    // A later turn's fan-out starts over at 1/N — old lines must not linger.
    emit('moa.progress', moaProgressPayload({ label: 'fresh', refs_done: 1, refs_total: 2 }))

    const text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/2 — fresh')
    expect(text).not.toContain('stale')
  })

  it('appends the aggregating marker on moa.phase and ignores unknown phases', () => {
    mountStream()

    emit('message.start', {})
    emit('moa.progress', moaProgressPayload({ label: 'model-a', refs_done: 1, refs_total: 1 }))
    emit('moa.phase', moaPhasePayload({ aggregator: null, phase: 'reference', refs_done: 1, refs_total: 1 }))
    expect(stream.reasoningText()).not.toContain('aggregating')

    emit('moa.phase', moaPhasePayload({ aggregator: 'agg-model', phase: 'aggregator', refs_done: 1, refs_total: 1 }))
    expect(stream.reasoningText()).toContain('MoA aggregating…')
  })

  it('a following moa.reference replaces the progress trail (self-cleaning)', () => {
    mountStream()

    emit('message.start', {})
    emit('moa.progress', moaProgressPayload({ label: 'model-a', refs_done: 1, refs_total: 1 }))
    emit('moa.phase', moaPhasePayload({ aggregator: null, phase: 'aggregator', refs_done: 1, refs_total: 1 }))
    emit('moa.reference', moaReferencePayload({ count: 1, index: 1, label: 'model-a', text: 'advice-a' }))

    const text = stream.reasoningText()
    expect(text).toContain('Reference 1/1 — model-a')
    expect(text).not.toContain('MoA refs')
    expect(text).not.toContain('aggregating')
  })
})
