import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $queuedPromptsBySession,
  claimConfirmedQueuedPrompt,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  isSteerableEntry,
  releaseConfirmedQueuedPrompt,
  resetFrozenQueuedTransportsForTests,
  resolveQueuedPromptTransport,
  simulateComposerQueueReloadForTests
} from './composer-queue'

const STORAGE = 'hermes.desktop.composerQueue.v1'

beforeEach(() => {
  window.localStorage.removeItem(STORAGE)
  $queuedPromptsBySession.set({})
  resetFrozenQueuedTransportsForTests()
})

afterEach(() => vi.restoreAllMocks())

describe('confirmed queue invariants', () => {
  it('keeps confirmed text out of redirect steering', () => {
    const entry = { attachments: [], text: 'confirmed instruction', confirmedExternal: true }
    expect(isSteerableEntry(entry)).toBe(false)
  })

  it('treats literal terminal tokens as confirmed plain text, not selected terminal output', () => {
    const entry = enqueueQueuedPrompt('session', {
      text: 'literal @terminal:shell:1',
      attachments: [],
      confirmedExternal: true
    })!

    expect(resolveQueuedPromptTransport(entry)).toEqual({ ok: true, transportText: 'literal @terminal:shell:1' })
    simulateComposerQueueReloadForTests()
    expect(resolveQueuedPromptTransport(getQueuedPrompts('session')[0])).toEqual({
      ok: true,
      transportText: 'literal @terminal:shell:1'
    })
  })

  it('holds durable dispatch intent after reload and cannot claim or auto-drain again', () => {
    const entry = enqueueQueuedPrompt('session', { text: 'confirmed', attachments: [], confirmedExternal: true })!
    expect(claimConfirmedQueuedPrompt('session', entry.id)).toBe(true)
    simulateComposerQueueReloadForTests()
    expect(getQueuedPrompts('session')[0].dispatchStarted).toBe(true)
    expect(claimConfirmedQueuedPrompt('session', entry.id)).toBe(false)
  })

  it('claim and release preserve another window queue without persisting frozen terminal payloads', () => {
    enqueueQueuedPrompt('terminal-session', {
      text: 'PRIVATE TERMINAL OUTPUT',
      displayText: '@terminal:shell:1',
      frozenTransport: 'PRIVATE TERMINAL OUTPUT',
      attachments: []
    })
    const entry = enqueueQueuedPrompt('session', { text: 'confirmed', attachments: [], confirmedExternal: true })!
    const live = JSON.parse(window.localStorage.getItem(STORAGE)!)
    live['other-window'] = [{ id: 'foreign', text: 'other window text', attachments: [], queuedAt: 1 }]
    window.localStorage.setItem(STORAGE, JSON.stringify(live))
    expect($queuedPromptsBySession.get()['other-window']).toBeUndefined()

    expect(claimConfirmedQueuedPrompt('session', entry.id)).toBe(true)
    expect(JSON.parse(window.localStorage.getItem(STORAGE)!)['other-window']).toEqual(live['other-window'])
    expect(window.localStorage.getItem(STORAGE)).not.toContain('PRIVATE TERMINAL OUTPUT')
    releaseConfirmedQueuedPrompt('session', entry.id)
    expect(getQueuedPrompts('session')[0].dispatchStarted).toBe(false)
    expect(JSON.parse(window.localStorage.getItem(STORAGE)!)['other-window']).toEqual(live['other-window'])
    expect(window.localStorage.getItem(STORAGE)).not.toContain('PRIVATE TERMINAL OUTPUT')
  })

  it('does not dispatch if durable storage refuses the intent write', () => {
    const entry = enqueueQueuedPrompt('session', { text: 'confirmed', attachments: [], confirmedExternal: true })!
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('storage unavailable')
    })
    expect(claimConfirmedQueuedPrompt('session', entry.id)).toBe(false)
    expect(getQueuedPrompts('session')[0].dispatchStarted).toBeUndefined()
  })
})
