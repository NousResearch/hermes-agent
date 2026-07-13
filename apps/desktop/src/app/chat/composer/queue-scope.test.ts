import { describe, expect, it } from 'vitest'

import { profileSessionKey } from '@/store/session-identity'

import {
  composerQueueKeys,
  queueScopeMigration,
  resolveComposerQueueScope
} from './queue-scope'

describe('composer queue key construction', () => {
  it('isolates the same runtime id across profiles', () => {
    const defaultKeys = composerQueueKeys('default', null, 'same-runtime')
    const workKeys = composerQueueKeys('work', null, 'same-runtime')

    expect(defaultKeys.queueRuntimeSessionKey).toBe(profileSessionKey('default', 'same-runtime'))
    expect(workKeys.queueRuntimeSessionKey).toBe(profileSessionKey('work', 'same-runtime'))
    expect(defaultKeys.queueRuntimeSessionKey).not.toBe(workKeys.queueRuntimeSessionKey)
  })

  it('uses the canonical runtime fallback before a stored id exists', () => {
    const keys = composerQueueKeys(' work ', null, 'runtime:["one"]')

    expect(resolveComposerQueueScope(keys)).toEqual({
      key: profileSessionKey('work', 'runtime:["one"]'),
      provenance: 'runtime',
      runtimeKey: profileSessionKey('work', 'runtime:["one"]')
    })
  })
})

describe('composer queue re-key provenance', () => {
  it('migrates runtime to stored for the same active runtime exactly once', () => {
    const runtime = resolveComposerQueueScope(composerQueueKeys('default', null, 'runtime-a'))
    const stored = resolveComposerQueueScope(composerQueueKeys('default', 'stored-a', 'runtime-a'))

    expect(queueScopeMigration(runtime, stored)).toEqual({
      fromKey: profileSessionKey('default', 'runtime-a'),
      toKey: profileSessionKey('default', 'stored-a')
    })
    expect(queueScopeMigration(stored, stored)).toBeNull()
  })

  it('does not migrate a stored A to stored B switch', () => {
    const storedA = resolveComposerQueueScope(composerQueueKeys('default', 'stored-a', 'runtime-a'))
    const storedB = resolveComposerQueueScope(composerQueueKeys('default', 'stored-b', 'runtime-b'))

    expect(queueScopeMigration(storedA, storedB)).toBeNull()
  })

  it('does not migrate runtime A to a different runtime B', () => {
    const runtimeA = resolveComposerQueueScope(composerQueueKeys('default', null, 'runtime-a'))
    const runtimeB = resolveComposerQueueScope(composerQueueKeys('default', null, 'runtime-b'))

    expect(queueScopeMigration(runtimeA, runtimeB)).toBeNull()
  })

  it('does not migrate the same bare identity across profiles', () => {
    const defaultRuntime = resolveComposerQueueScope(composerQueueKeys('default', null, 'same'))
    const workStored = resolveComposerQueueScope(composerQueueKeys('work', 'same', 'same'))
    const defaultStored = resolveComposerQueueScope(composerQueueKeys('default', 'same', 'same'))

    expect(queueScopeMigration(defaultRuntime, workStored)).toBeNull()
    expect(queueScopeMigration(defaultStored, workStored)).toBeNull()
  })

  it('keeps a stored queue stable when the runtime rotates', () => {
    const before = resolveComposerQueueScope(composerQueueKeys('default', 'stored', 'runtime-a'))
    const after = resolveComposerQueueScope(composerQueueKeys('default', 'stored', 'runtime-b'))

    expect(before?.key).toBe(after?.key)
    expect(queueScopeMigration(before, after)).toBeNull()
  })
})
