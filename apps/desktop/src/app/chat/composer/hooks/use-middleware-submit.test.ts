import { describe, expect, it } from 'vitest'

import { composerSendOrigin } from './use-middleware-submit'

describe('composerSendOrigin', () => {
  it('reports typed only for the composer Enter path', () => {
    expect(composerSendOrigin('main', { typedDraft: true, composerScope: 'a' })).toEqual({
      kind: 'typed',
      target: 'main',
      composerScope: 'a'
    })
    expect(composerSendOrigin('main', { composerScope: 'a', attachments: [] }).kind).toBe('other')
    expect(composerSendOrigin('main').kind).toBe('other')
  })

  it('never classifies machine-driven sends as typed, even when they carry the typed flag', () => {
    expect(composerSendOrigin('main', { typedDraft: true, fromQueue: true }).kind).toBe('queued')
    expect(composerSendOrigin('main', { typedDraft: true, voiceTurn: true }).kind).toBe('voice')
    expect(composerSendOrigin('main', { typedDraft: true, surface: 'voice-live' }).kind).toBe('voice')
    expect(composerSendOrigin('main', { typedDraft: true, displayKind: 'hidden' }).kind).toBe('hidden')
  })

  it('keeps a fresh chat (null scope) distinct from a non-composer send (no scope)', () => {
    expect(composerSendOrigin('main', { typedDraft: true, composerScope: null }).composerScope).toBeNull()
    expect('composerScope' in composerSendOrigin('main', { fromQueue: true })).toBe(false)
  })
})
