import { describe, expect, it } from 'vitest'

import { deriveBusySubmitState } from './busy-submit-state'

// A busy draft's routing: what the Send button does while a turn is running.
// Steer/redirect injects into the live turn and therefore cannot carry an image
// (a tool result holds text, not pixels). Interrupt does not inject anything —
// it ends the turn and lets the draft run as the next one — so attachments are
// no reason to demote it to the queue (NousResearch/hermes-agent#96759).
const base = {
  attachmentCount: 0,
  blockingPrompt: false,
  busy: true,
  busyInputMode: 'interrupt' as const,
  compacting: false,
  hasComposerPayload: true,
  hasSteerHandler: true,
  isSteerableText: true,
}

describe('deriveBusySubmitState — busy routing', () => {
  it('a text-only draft interrupts the live turn', () => {
    const state = deriveBusySubmitState(base)

    expect(state.canSteer).toBe(true)
    expect(state.busyAction).toBe('interrupt')
  })

  it('a draft carrying an attachment still interrupts instead of queueing', () => {
    const state = deriveBusySubmitState({ ...base, attachmentCount: 1 })

    // Steer stays text-only — an image cannot ride a redirect.
    expect(state.canSteer).toBe(false)
    // …but the interrupt path carries no payload, so the image is no obstacle.
    expect(state.canInterrupt).toBe(true)
    expect(state.busyAction).toBe('interrupt')
  })

  it('an attachment-only draft interrupts too', () => {
    const state = deriveBusySubmitState({
      ...base,
      attachmentCount: 2,
      isSteerableText: false
    })

    expect(state.canInterrupt).toBe(true)
    expect(state.busyAction).toBe('interrupt')
  })
})

describe('deriveBusySubmitState — what still waits', () => {
  it('queue mode is the user asking to wait, attachments or not', () => {
    const state = deriveBusySubmitState({ ...base, attachmentCount: 1, busyInputMode: 'queue' })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('queue')
  })

  it('an unloaded busy policy never turns into an interrupt', () => {
    const state = deriveBusySubmitState({ ...base, attachmentCount: 1, busyInputMode: null })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('queue')
  })

  it('a parked approval prompt keeps the draft queued', () => {
    const state = deriveBusySubmitState({ ...base, attachmentCount: 1, blockingPrompt: true })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('queue')
  })

  it('in-flight compaction keeps the draft queued', () => {
    const state = deriveBusySubmitState({ ...base, attachmentCount: 1, compacting: true })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('queue')
  })

  it('a bare slash command still queues rather than claiming the interrupt', () => {
    const state = deriveBusySubmitState({ ...base, isSteerableText: false })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('queue')
  })

  it('an empty composer is the Stop button', () => {
    const state = deriveBusySubmitState({ ...base, hasComposerPayload: false, isSteerableText: false })

    expect(state.canInterrupt).toBe(false)
    expect(state.busyAction).toBe('stop')
  })
})
