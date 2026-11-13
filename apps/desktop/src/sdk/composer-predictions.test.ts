import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $composerPredictions,
  bindComposerPredictions,
  MAX_PREDICTION_CANDIDATES,
  MAX_PREDICTION_TEXT_LENGTH
} from '@/app/chat/composer/predictions'
import type { ClientSessionState } from '@/app/types'
import type { HermesConnection } from '@/global'
import { stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $activeGatewayProfile, $activeProfile } from '@/store/profile'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { composerHost } from './composer'

import type { ComposerPredictionHandle, ComposerPredictionScope, ComposerPredictionsPayload } from './index'

const scope: ComposerPredictionScope = { sessionId: 'runtime-a', profile: 'default', connectionId: 'local' }
const payload: ComposerPredictionsPayload = { ...scope, suggestions: ['First', 'Second'] }

const connection: HermesConnection = {
  connectionId: 'local',
  baseUrl: 'http://localhost',
  wsUrl: 'ws://localhost',
  token: '',
  logs: [],
  isFullscreen: false,
  nativeOverlayWidth: 0,
  windowButtonPosition: null
}

let binding: ReturnType<typeof bindComposerPredictions>

beforeEach(() => {
  $activeProfile.set(scope.profile)
  $activeGatewayProfile.set(scope.profile)
  $connection.set(connection)
  $activeSessionId.set(scope.sessionId)
  $selectedStoredSessionId.set('stored-a')
  // Only the resolver's durable/runtime mapping participates in this bridge test.
  $sessionStates.set({ 'runtime-a': { storedSessionId: 'stored-a' } as ClientSessionState })
  binding = bindComposerPredictions(scope, () => true)
})
afterEach(() => {
  binding.dispose()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
})

describe('public composer prediction bridge', () => {
  it('uses the live routed profile while the HTTP display profile is stale', () => {
    const previous = composerHost.setPredictions(payload)!
    $activeGatewayProfile.set('work')
    expect($activeProfile.get()).toBe('default')
    expect($composerPredictions.get()).toBeNull()
    expect(composerHost.clearPredictions(previous)).toBe(false)
    binding = bindComposerPredictions({ ...scope, profile: 'work' }, () => true)
    expect(composerHost.setPredictions(payload)).toBeNull()
    const current = composerHost.setPredictions({ ...payload, profile: 'work' })
    expect(current).not.toBeNull()
    // A late HTTP display refresh is not a new route and cannot revoke it.
    $activeProfile.set('work')
    expect($composerPredictions.get()?.handle).toBe(current)
  })

  it('acknowledges runtime and durable ids and only clears the exact current receipt', () => {
    const first = composerHost.setPredictions(payload)!
    expect(first).not.toBeNull()
    const second: ComposerPredictionHandle | null = composerHost.setPredictions({ ...payload, sessionId: 'stored-a' })
    expect(second).not.toBeNull()
    expect(second).not.toBe(first)
    expect(composerHost.clearPredictions(first)).toBe(false)
    expect(composerHost.clearPredictions({ ...second! })).toBe(false)
    binding.cycle()
    expect($composerPredictions.get()?.index).toBe(1)
    expect(composerHost.clearPredictions(second!)).toBe(true)
    expect(composerHost.clearPredictions(second!)).toBe(false)
    expect($composerPredictions.get()).toBeNull()
  })

  it('fails closed for tiles, absent receivers, new drafts and unknown session ids', () => {
    for (const sessionId of ['tile:stored-a', 'new', 'unmounted', 'runtime-b']) {
      expect(composerHost.setPredictions({ ...payload, sessionId })).toBeNull()
    }

    binding.dispose()
    expect(composerHost.setPredictions(payload)).toBeNull()
  })

  it('does not let stale receiver cleanup or actions affect a replacement receiver', () => {
    const first = composerHost.setPredictions(payload)!
    const oldBinding = binding
    binding = bindComposerPredictions(scope, () => true)
    const second = composerHost.setPredictions(payload)!
    expect(composerHost.clearPredictions(first)).toBe(false)
    oldBinding.clear()
    oldBinding.cycle()
    oldBinding.dispose()
    expect($composerPredictions.get()?.handle).toBe(second)
    expect($composerPredictions.get()?.index).toBe(0)
    expect(composerHost.clearPredictions(second)).toBe(true)
  })

  it('fails closed while selected durable and active runtime identities disagree', () => {
    $selectedStoredSessionId.set('stored-b')
    binding = bindComposerPredictions(scope, () => true)
    expect(composerHost.setPredictions({ ...payload, sessionId: 'stored-b' })).toBeNull()
    expect($composerPredictions.get()).toBeNull()
  })

  it('keeps session state and persisted drafts unchanged when publishing', () => {
    stashSessionDraft('stored-a', 'Untouched persisted draft', [])
    const states = $sessionStates.get()
    const storage = { ...localStorage }
    const receipt = composerHost.setPredictions(payload)!
    expect(receipt).not.toBeNull()
    binding.cycle()
    expect(composerHost.clearPredictions(receipt)).toBe(true)
    expect($sessionStates.get()).toBe(states)
    expect(takeSessionDraft('stored-a').text).toBe('Untouched persisted draft')
    expect({ ...localStorage }).toEqual(storage)
    stashSessionDraft('stored-a', '', [])
  })

  it.each(['profile', 'connection', 'session', 'selected', 'disconnect', 'endpoint'])(
    'revokes publications and receivers synchronously on %s changes',
    edge => {
      const receipt = composerHost.setPredictions(payload)!

      if (edge === 'profile') {
        $activeGatewayProfile.set('other')
      }

      if (edge === 'connection') {
        $connection.set({ ...connection, connectionId: 'remote' })
      }

      if (edge === 'session') {
        $activeSessionId.set('runtime-b')
      }

      if (edge === 'selected') {
        $selectedStoredSessionId.set('stored-b')
      }

      if (edge === 'disconnect') {
        $connection.set(null)
      }

      if (edge === 'endpoint') {
        $connection.set({ ...connection, baseUrl: 'https://other.test' })
      }

      expect($composerPredictions.get()).toBeNull()
      expect(composerHost.clearPredictions(receipt)).toBe(false)
      expect(composerHost.setPredictions(payload)).toBeNull()
      $activeGatewayProfile.set(scope.profile)
      $connection.set(connection)
      $activeSessionId.set(scope.sessionId)
      $selectedStoredSessionId.set('stored-a')
      // A→B→A does not revive the revoked receiver.
      expect(composerHost.setPredictions(payload)).toBeNull()
      binding = bindComposerPredictions(scope, () => true)
      expect(composerHost.setPredictions(payload)).not.toBeNull()
    }
  )

  it('rejects the old owner when profile or connection reuses the same session ids', () => {
    const oldReceipt = composerHost.setPredictions(payload)!
    $activeGatewayProfile.set('other')
    $connection.set({ ...connection, connectionId: 'remote' })
    const next = { ...scope, profile: 'other', connectionId: 'remote' }
    binding = bindComposerPredictions(next, () => true)
    expect(composerHost.setPredictions(payload)).toBeNull()
    expect(composerHost.setPredictions({ ...payload, profile: 'other' })).toBeNull()
    expect(composerHost.setPredictions({ ...payload, connectionId: 'remote' })).toBeNull()
    const receipt = composerHost.setPredictions({ ...next, suggestions: ['New owner'] })!
    expect(receipt).not.toBeNull()
    expect(composerHost.clearPredictions(oldReceipt)).toBe(false)
    expect($composerPredictions.get()?.suggestions).toEqual(['New owner'])
  })

  it('does not treat window geometry changes as authority changes', () => {
    const receipt = composerHost.setPredictions(payload)!
    $connection.set({ ...connection, isFullscreen: true, nativeOverlayWidth: 42 })
    expect($composerPredictions.get()?.handle).toBe(receipt)
    expect(composerHost.clearPredictions(receipt)).toBe(true)
  })

  it.each([
    null,
    undefined,
    false,
    'session',
    {},
    [],
    { ...payload, sessionId: null },
    { ...payload, sessionId: ' runtime-a ' },
    { ...payload, profile: null },
    { ...payload, profile: '' },
    { ...payload, connectionId: null },
    { ...payload, connectionId: '' },
    { ...payload, suggestions: null },
    { ...payload, suggestions: 'text' },
    { ...payload, suggestions: [null] },
    { ...payload, suggestions: [123] },
    { ...payload, suggestions: [] },
    { ...payload, suggestions: ['  ', '\n'] },
    { ...payload, suggestions: Array(MAX_PREDICTION_CANDIDATES + 1).fill('candidate') },
    { ...payload, suggestions: ['x'.repeat(MAX_PREDICTION_TEXT_LENGTH + 1)] },
    { ...payload, sessionId: 'x'.repeat(513) }
  ])('rejects malformed or excessive JS input without erasing a publication: %j', input => {
    const receipt = composerHost.setPredictions(payload)!
    expect(composerHost.setPredictions(input as ComposerPredictionsPayload)).toBeNull()
    expect($composerPredictions.get()?.handle).toBe(receipt)
  })

  it.each([null, undefined, 'runtime-a', {}, { publication: Symbol('fake') }])(
    'rejects malformed cleanup handles: %j',
    input => {
      const receipt = composerHost.setPredictions(payload)!
      expect(composerHost.clearPredictions(input as ComposerPredictionHandle)).toBe(false)
      expect($composerPredictions.get()?.handle).toBe(receipt)
    }
  )

  it('preserves full valid text at the bounds, copies input and deduplicates exact matches', () => {
    const full = '  完整\n' + 'x'.repeat(MAX_PREDICTION_TEXT_LENGTH - 7) + '  '
    expect(full.length).toBe(MAX_PREDICTION_TEXT_LENGTH)

    const suggestions = [
      full,
      full,
      '  ',
      ...Array.from({ length: MAX_PREDICTION_CANDIDATES - 3 }, (_, i) => `Message ${i}`)
    ]

    expect(composerHost.setPredictions({ ...payload, suggestions })).not.toBeNull()
    suggestions[0] = 'Changed after publishing'
    expect($composerPredictions.get()?.suggestions[0]).toBe(full)
    expect($composerPredictions.get()?.suggestions).toHaveLength(MAX_PREDICTION_CANDIDATES - 2)
  })
})
