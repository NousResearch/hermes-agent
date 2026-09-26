import { describe, expect, it } from 'vitest'

import { normalizeBusyInputMode, resolveBusyComposerAction } from './busy-input-mode'

const base = {
  busy: true,
  canCorrect: true,
  compacting: false,
  hasPayload: true,
  blockingPrompt: false,
  mode: 'interrupt' as const
}

describe('normalizeBusyInputMode', () => {
  it('passes through the canonical values', () => {
    expect(normalizeBusyInputMode('interrupt')).toBe('interrupt')
    expect(normalizeBusyInputMode('queue')).toBe('queue')
    expect(normalizeBusyInputMode('steer')).toBe('steer')
  })

  it('trims and lowercases', () => {
    expect(normalizeBusyInputMode('  Steer ')).toBe('steer')
  })

  it('falls back to interrupt for unknown or malformed values', () => {
    expect(normalizeBusyInputMode('redirect')).toBe('interrupt')
    expect(normalizeBusyInputMode('')).toBe('interrupt')
    expect(normalizeBusyInputMode(undefined)).toBe('interrupt')
    expect(normalizeBusyInputMode(null)).toBe('interrupt')
    expect(normalizeBusyInputMode(3)).toBe('interrupt')
  })
})

describe('resolveBusyComposerAction', () => {
  it('stops when idle', () => {
    expect(resolveBusyComposerAction({ ...base, busy: false })).toBe('stop')
  })

  it('redirects in interrupt mode', () => {
    expect(resolveBusyComposerAction({ ...base })).toBe('redirect')
  })

  it('steers at the tool boundary in steer mode', () => {
    expect(resolveBusyComposerAction({ ...base, mode: 'steer' })).toBe('steer')
  })

  it('queues in queue mode', () => {
    expect(resolveBusyComposerAction({ ...base, mode: 'queue' })).toBe('queue')
  })

  it('stops in queue mode with an empty composer', () => {
    expect(resolveBusyComposerAction({ ...base, mode: 'queue', hasPayload: false })).toBe('stop')
  })

  it('falls back to queue when the payload cannot ride a correction', () => {
    expect(resolveBusyComposerAction({ ...base, canCorrect: false })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, mode: 'steer', canCorrect: false })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, canCorrect: false, hasPayload: false })).toBe('stop')
  })

  it('never corrects a turn parked on a blocking prompt', () => {
    expect(resolveBusyComposerAction({ ...base, blockingPrompt: true })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, blockingPrompt: true, mode: 'steer' })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, blockingPrompt: true, hasPayload: false })).toBe('stop')
  })

  it('never corrects while compacting', () => {
    expect(resolveBusyComposerAction({ ...base, compacting: true })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, compacting: true, mode: 'queue' })).toBe('queue')
    expect(resolveBusyComposerAction({ ...base, compacting: true, hasPayload: false })).toBe('stop')
  })
})
