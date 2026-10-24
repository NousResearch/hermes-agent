import { describe, expect, it } from 'vitest'

import { PLAYER_CONTEXT_TOKEN_TTL_MS } from '../../src/lib/player-context-contract'

import {
  PlayerContextRegistry,
  playerControlGenerationIsCurrent,
  playerOwnerMatches,
  youtubeTargetFromUrl
} from './context-capabilities'

const scope = { connectionId: 'local', profileId: 'default', sessionId: 'session-1' }

const target = {
  source: 'youtube' as const,
  videoId: 'abcdefghijk',
  canonicalUrl: 'https://www.youtube.com/watch?v=abcdefghijk',
  title: { text: 'Untrusted title', trust: 'untrusted-page' as const },
  capturedTimeSeconds: 12,
  adStatus: 'unknown' as const
}

describe('player context capabilities', () => {
  it('binds opaque contexts to owner, linked chat, and a five-minute expiry', () => {
    const registry = new PlayerContextRegistry()
    const issued = registry.issue({ ownerId: 9, linkId: 'native-link', scope, generation: 4, target }, 1000)
    expect(issued.token).not.toContain('abcdefghijk')
    expect(issued.expiresAtMs).toBe(1000 + PLAYER_CONTEXT_TOKEN_TTL_MS)
    expect(registry.resolve(issued.token, 9, 'native-link', scope, 1001)?.generation).toBe(4)
    expect(registry.resolve(issued.token, 10, 'native-link', scope, 1001)).toBeNull()
    expect(registry.resolve(issued.token, 9, 'other-link', scope, 1001)).toBeNull()
    expect(registry.resolve(issued.token, 9, 'native-link', { ...scope, sessionId: 'other' }, 1001)).toBeNull()
    expect(registry.resolve(issued.token, 9, 'native-link', scope, 1001)).not.toBeNull()
    expect(registry.resolve(issued.token, 9, 'native-link', scope, issued.expiresAtMs)).toBeNull()
  })

  it('preserves the immutable captured target across navigation generations', () => {
    const registry = new PlayerContextRegistry()
    const issued = registry.issue({ ownerId: 9, linkId: 'native-link', scope, generation: 1, target }, 10)
    const resolved = registry.resolve(issued.token, 9, 'native-link', scope, 11)!
    expect(resolved.target).toEqual(target)
    expect(resolved.generation).toBe(1)
    expect(resolved.target).not.toBe(target)
    expect(playerControlGenerationIsCurrent(resolved, 1)).toBe(true)
    expect(playerControlGenerationIsCurrent(resolved, 2)).toBe(false)
  })

  it('creates a media target only from exact HTTPS YouTube watch identity', () => {
    const metadata = { title: null, capturedTimeSeconds: null, adStatus: 'unknown' as const }
    expect(
      youtubeTargetFromUrl('youtube', 'https://www.youtube.com/watch?v=abcdefghijk&list=playlist', metadata)
    ).toMatchObject({
      source: 'youtube',
      videoId: 'abcdefghijk',
      canonicalUrl: 'https://www.youtube.com/watch?v=abcdefghijk'
    })

    for (const [source, url] of [
      ['tv', 'https://www.youtube.com/watch?v=abcdefghijk'],
      ['youtube', 'http://www.youtube.com/watch?v=abcdefghijk'],
      ['youtube', 'https://youtube.com.evil.example/watch?v=abcdefghijk'],
      ['youtube', 'https://www.youtube.com/embed/abcdefghijk'],
      ['youtube', 'https://www.youtube.com/watch?v=short']
    ]) {
      expect(youtubeTargetFromUrl(source, url, metadata)).toBeNull()
    }
  })

  it('does not let another renderer replace or clear a player owner binding', () => {
    expect(playerOwnerMatches(9, null, undefined, 9)).toBe(true)
    expect(playerOwnerMatches(10, 9)).toBe(false)
    expect(playerOwnerMatches(10, null, 9, undefined)).toBe(false)
  })
})
