import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { claimSideTask } from '@/contrib/side-task-claims'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

let stream: MessageStreamHarness

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload']) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

// A plugin that renders a /background or /btw answer on its own surface claims
// the task id; the transcript must then stay clean for THAT task only, and the
// answer must never be lost to a claim nobody consumes.
describe('side-task claims', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  it.each([
    ['background.complete', { task_id: 'bg_aa11bb', text: 'done' }],
    ['btw.complete', { task_id: 'btw_aa11bb', question: 'q', text: 'done' }]
  ] as const)('%s: a claimed task stays out of the transcript, an unclaimed one lands', (type, payload) => {
    claimSideTask(payload.task_id)
    emit(type, payload)
    expect(stream.state(SID).messages).toHaveLength(0)

    emit(type, { ...payload, task_id: `${payload.task_id}_other` })
    expect(stream.state(SID).messages).toHaveLength(1)
  })

  it('a claim is consumed by its delivery, so a repeated id falls back to the transcript', () => {
    claimSideTask('bg_once01')
    emit('background.complete', { task_id: 'bg_once01', text: 'first' })
    emit('background.complete', { task_id: 'bg_once01', text: 'second' })

    expect(stream.text()).toBe('[bg bg_once01]\nsecond')
  })

  it('a claim released before delivery (plugin unloaded) returns the answer to the transcript', () => {
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('side-window', dispose => void disposers.push(dispose))

    ctx.claimSideTask('bg_gone01')
    disposers.forEach(dispose => dispose())
    emit('background.complete', { task_id: 'bg_gone01', text: 'still here' })

    expect(stream.text()).toBe('[bg bg_gone01]\nstill here')
  })
})
