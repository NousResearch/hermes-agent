import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useMessageStream moa.progress / moa.phase surfacing', () => {
  it('shows refs k/n lines in the reasoning block as references complete', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 3 })
    emit('moa.progress', { label: 'model-b', refs_done: 2, refs_total: 3 })

    const text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/3 — model-a')
    expect(text).toContain('MoA refs 2/3 — model-b')
  })

  it('replaces a waiting heartbeat instead of appending transcript noise', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', {
      elapsed_seconds: 5,
      label: 'waiting on model-c · 5s',
      refs_done: 2,
      refs_total: 3,
      state: 'waiting'
    })
    emit('moa.progress', {
      elapsed_seconds: 10,
      label: 'waiting on model-c · 10s',
      refs_done: 2,
      refs_total: 3,
      state: 'waiting'
    })

    const text = stream.reasoningText()
    expect(text).toContain('MoA refs 2/3 — waiting on model-c · 10s')
    expect(text).not.toContain('waiting on model-c · 5s')
  })

  it('keeps completed references while replacing and settling the waiting line', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', {
      elapsed_seconds: 5,
      label: 'waiting on model-c · 5s',
      refs_done: 0,
      refs_total: 3,
      state: 'waiting'
    })
    emit('moa.progress', { label: 'moa:nested', refs_done: 1, refs_total: 3, state: 'completed' })
    emit('moa.progress', {
      elapsed_seconds: 10,
      label: 'waiting on model-c · 10s',
      refs_done: 1,
      refs_total: 3,
      state: 'waiting'
    })
    emit('moa.progress', { label: 'model-a', refs_done: 2, refs_total: 3, state: 'completed' })
    emit('moa.progress', {
      elapsed_seconds: 15,
      label: 'waiting on model-b · 15s',
      refs_done: 2,
      refs_total: 3,
      state: 'waiting'
    })
    emit('moa.progress', {
      elapsed_seconds: 20,
      label: 'waiting on model-b · 20s',
      refs_done: 2,
      refs_total: 3,
      state: 'waiting'
    })

    let text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/3 — moa:nested')
    expect(text).toContain('MoA refs 2/3 — model-a')
    expect(text).toContain('waiting on model-b · 20s')
    expect(text).not.toContain('waiting on model-b · 15s')
    expect(text).not.toContain('waiting on model-c')

    emit('moa.progress', { label: 'model-b', refs_done: 3, refs_total: 3, state: 'completed' })
    text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/3 — moa:nested')
    expect(text).toContain('MoA refs 2/3 — model-a')
    expect(text).toContain('MoA refs 3/3 — model-b')
    expect(text).not.toContain('waiting on')
  })

  it('restarts the progress block on the first ref of a new fan-out', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'stale', refs_done: 1, refs_total: 2 })
    emit('moa.progress', { label: 'stale-2', refs_done: 2, refs_total: 2 })
    // A later turn's fan-out starts over at 1/N — old lines must not linger.
    emit('moa.progress', { label: 'fresh', refs_done: 1, refs_total: 2 })

    const text = stream.reasoningText()
    expect(text).toContain('MoA refs 1/2 — fresh')
    expect(text).not.toContain('stale')
  })

  it('appends the aggregating marker on moa.phase and ignores unknown phases', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 1 })
    emit('moa.phase', { phase: 'reference', refs_done: 1, refs_total: 1 })
    expect(stream.reasoningText()).not.toContain('aggregating')

    emit('moa.phase', { aggregator: 'agg-model', phase: 'aggregator', refs_done: 1, refs_total: 1 })
    expect(stream.reasoningText()).toContain('MoA aggregating…')
  })

  it('a following moa.reference replaces the progress trail (self-cleaning)', () => {
    mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 1 })
    emit('moa.phase', { phase: 'aggregator', refs_done: 1, refs_total: 1 })
    emit('moa.reference', { count: 1, index: 1, label: 'model-a', text: 'advice-a' })

    const text = stream.reasoningText()
    expect(text).toContain('Reference 1/1 — model-a')
    expect(text).not.toContain('MoA refs')
    expect(text).not.toContain('aggregating')
  })
})
