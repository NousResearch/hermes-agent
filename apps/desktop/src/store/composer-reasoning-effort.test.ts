import { beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '@/test/deferred'

import {
  $composerReasoningEffort,
  rescopeComposerReasoningEffort,
  setComposerReasoningEffort
} from './composer-reasoning-effort'
import { applyModelPreset } from './model-presets'
import { $activeSessionId, setComposerSelectionOwner, setCurrentReasoningEffort } from './session'
import { dropTilesForProfile, migrateTilesForProfile } from './session-states'

beforeEach(() => {
  $activeSessionId.set(null)

  for (const connection of ['local', 'source-a', 'source-b']) {
    for (const profile of ['default', 'renamed']) {
      setComposerSelectionOwner(connection, profile)
      setComposerReasoningEffort('')
    }
  }
})

describe('composer effort ownership', () => {
  it('persists explicit picks across module reload, and follows local rename and exact removal', async () => {
    setComposerSelectionOwner('local', 'default')
    setComposerReasoningEffort('none')
    setComposerSelectionOwner('source-a', 'default')
    setComposerReasoningEffort('high')
    setComposerSelectionOwner('source-b', 'default')
    expect($composerReasoningEffort.get()).toBe('')
    migrateTilesForProfile('default', 'renamed')
    setComposerSelectionOwner('local', 'renamed')
    expect($composerReasoningEffort.get()).toBe('none')
    setComposerSelectionOwner('source-a', 'default')
    expect($composerReasoningEffort.get()).toBe('high')
    dropTilesForProfile('default', { connectionId: 'source-b', profile: 'default' })
    expect($composerReasoningEffort.get()).toBe('high')
    dropTilesForProfile('default', { connectionId: 'source-a', profile: 'default' })
    expect($composerReasoningEffort.get()).toBe('')
    vi.resetModules()
    const reloaded = await import('./composer-reasoning-effort')
    reloaded.rescopeComposerReasoningEffort('.registry.local.renamed', 'renamed')
    expect(reloaded.$composerReasoningEffort.get()).toBe('none')
    rescopeComposerReasoningEffort(null)
    setComposerReasoningEffort('low')
    setComposerSelectionOwner('local', 'renamed')
    expect($composerReasoningEffort.get()).toBe('none')
    setComposerSelectionOwner('local', 'default')
    dropTilesForProfile('renamed')
    setComposerSelectionOwner('local', 'renamed')
    expect($composerReasoningEffort.get()).toBe('')
  })

  it('rolls back overlapping failed edits to confirmed intent rather than a runtime mirror', async () => {
    setComposerSelectionOwner('source-a', 'default')
    setComposerReasoningEffort('none')
    setCurrentReasoningEffort('low')
    const write = deferred<void>()
    const request = vi.fn(() => write.promise as Promise<never>)
    const ctx = { failMessage: 'failed', request, sessionId: 'runtime' }
    const first = applyModelPreset({ effort: 'high' }, ctx)
    const second = applyModelPreset({ effort: 'medium' }, ctx)
    write.reject(new Error('fixture failure'))
    await Promise.all([first, second])
    expect($composerReasoningEffort.get()).toBe('none')
    expect(request).toHaveBeenCalledTimes(2)
  })
})
