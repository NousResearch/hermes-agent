import { afterEach, expect, it } from 'vitest'

import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { resolveActiveTranscriptSession } from './use-background-sync'

afterEach(() => {
  $activeSessionId.set(null)
  $sessionTiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests()
})

it('does not promote a different runtime tile into the active transcript owner', () => {
  const ownerRoute = { connectionId: 'local', profile: 'other-profile', mode: 'local' as const }
  setSessions([{ id: 'shared', profile: 'default', source: 'desktop' } as never])
  $sessionTiles.set([{ storedSessionId: 'shared', runtimeId: 'other-runtime', ownerRoute }])

  expect(resolveActiveTranscriptSession('shared', 'active-runtime')).toEqual({ profile: 'default' })
})

it('does not treat an ownerless active tile as corroboration for a stale hint', () => {
  setSessions([{ id: 'shared', profile: 'default', source: 'desktop' } as never])
  setSessionOwnerHint('shared', { connectionId: 'stale-connection', profile: 'stale-profile', mode: 'remote' })
  $sessionTiles.set([{ storedSessionId: 'shared', runtimeId: 'active-runtime' }])

  expect(resolveActiveTranscriptSession('shared', 'active-runtime')).toEqual({ profile: 'default' })
})
