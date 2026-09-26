import { afterEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection, forgetSessionOwnerHintsForSession, setSessionOwnerHint } from '@/store/session'

import { previewOwnerIsAmbient } from './preview-owner'

const sessionId = 'preview-foreign-owner-test'
const previousProfile = $activeGatewayProfile.get()
const previousConnection = $connection.get()
const previousSession = $activeSessionId.get()

afterEach(() => {
  forgetSessionOwnerHintsForSession(sessionId)
  $activeGatewayProfile.set(previousProfile)
  $connection.set(previousConnection)
  $activeSessionId.set(previousSession)
})

describe('previewOwnerIsAmbient', () => {
  it('refuses to externalize a visible session from another registered connection', () => {
    $activeSessionId.set(sessionId)
    $activeGatewayProfile.set('default')
    $connection.set({ connectionId: 'local-machine', mode: 'local' } as never)
    setSessionOwnerHint(sessionId, { connectionId: 'remote-machine', profile: 'default' })

    expect(previewOwnerIsAmbient(sessionId)).toBe(false)
  })

  it('allows the active connection and profile, but not a different profile on it', () => {
    $activeGatewayProfile.set('default')
    $connection.set({ connectionId: 'remote-machine', mode: 'remote' } as never)
    setSessionOwnerHint(sessionId, { connectionId: 'remote-machine', profile: 'default' })

    expect(previewOwnerIsAmbient(sessionId)).toBe(true)
    $activeGatewayProfile.set('other')
    expect(previewOwnerIsAmbient(sessionId)).toBe(false)
  })
})
