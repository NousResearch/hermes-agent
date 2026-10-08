import { afterEach, expect, test, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '../scope'

import { composerSubmissionSlot, prepareComposerSubmission, rehomeComposerSubmission, retireComposerSubmission } from './composer-submission-journal'

afterEach(() => { sessionStorage.clear(); vi.restoreAllMocks() })

function saveWindowStorage() {
  return Array.from({ length: sessionStorage.length }, (_, index) => {
    const key = sessionStorage.key(index)!

    return [key, sessionStorage.getItem(key)!] as const
  })
}

test('separate windows keep distinct operations and a serialized window restores the same ID after module reload', async () => {
  const scope = { ...MAIN_COMPOSER_SCOPE, connectionId: 'server', profile: 'work' }
  const slot = composerSubmissionSlot('session', scope, 'persisted-pane')
  const first = prepareComposerSubmission(slot, 'identical text', [], new Set())
  const firstWindow = saveWindowStorage()
  sessionStorage.clear()
  const second = prepareComposerSubmission(slot, 'identical text', [], new Set())
  expect(second).not.toBe(first)
  const secondWindow = saveWindowStorage()
  // A fresh module instance has no component, ref, or in-memory ID to recover.
  vi.resetModules()
  const reloaded = await import('./composer-submission-journal')
  sessionStorage.clear()

  for (const [key, value] of firstWindow) { sessionStorage.setItem(key, value) }
  expect(reloaded.prepareComposerSubmission(slot, 'identical text', [], new Set())).toBe(first)
  sessionStorage.clear()

  for (const [key, value] of secondWindow) { sessionStorage.setItem(key, value) }
  expect(reloaded.prepareComposerSubmission(slot, 'identical text', [], new Set())).toBe(second)
})

test('fresh-session assignment keeps the retained operation and an earlier ACK never removes a newer send', () => {
  const slot = composerSubmissionSlot('__new__:draft', MAIN_COMPOSER_SCOPE, 'persisted-pane')
  const first = prepareComposerSubmission(slot, 'first text', [], new Set())
  const assigned = rehomeComposerSubmission(slot, 'stored-session', first)
  expect(sessionStorage.getItem(slot)).toBeNull()
  expect(prepareComposerSubmission(assigned, 'first text', [], new Set())).toBe(first)
  const next = prepareComposerSubmission(assigned, 'new text', [], new Set())
  retireComposerSubmission(assigned, first)
  expect(prepareComposerSubmission(assigned, 'new text', [], new Set())).toBe(next)
  retireComposerSubmission(assigned, next)
  expect(sessionStorage.getItem(assigned)).toBeNull()
})
