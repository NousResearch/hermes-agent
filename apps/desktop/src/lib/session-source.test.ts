import { describe, expect, it } from 'vitest'

import { isMessagingSource, MESSAGING_SESSION_SOURCE_IDS, sessionSourceSearchTerms } from './session-source'

// #67794: every gateway platform gets its own sidebar section, not just the
// whitelisted ones. isMessagingSource() is the section gate — an
// exclusion-based test, so a platform the catalogs have never heard of still
// groups into its own collapsible section instead of falling into the generic
// Sessions list. These asserts pin that contract; a reversion to a fixed
// platform whitelist regresses the feature with no other test failure.
describe('messaging source classification', () => {
  it('treats an unrecognized platform source as messaging (own sidebar section)', () => {
    expect(isMessagingSource('irc')).toBe(true)
    expect(isMessagingSource('google_chat')).toBe(true)
    expect(isMessagingSource('teams')).toBe(true)
  })

  it('keeps registered platforms messaging (photon regression, #46761)', () => {
    expect(isMessagingSource('photon')).toBe(true)
    expect(isMessagingSource('telegram')).toBe(true)
  })

  it('is case/space insensitive on the source id', () => {
    expect(isMessagingSource('PHOTON')).toBe(true)
    expect(isMessagingSource('  photon ')).toBe(true)
  })

  it('does not flag local, editor, or automation sources as messaging', () => {
    for (const source of ['cli', 'tui', 'desktop', 'acp', 'gateway', 'local', 'oneshot', 'kanban', 'cron', 'subagent', 'tool']) {
      expect(isMessagingSource(source)).toBe(false)
    }

    expect(isMessagingSource(null)).toBe(false)
    expect(isMessagingSource(undefined)).toBe(false)
    expect(isMessagingSource('')).toBe(false)
  })

  it('exposes the iMessage/messages search aliases so Photon sessions are findable', () => {
    const terms = sessionSourceSearchTerms('photon')
    expect(terms).toContain('imessage')
    expect(terms).toContain('messages')
  })

  it('keeps the known-platform metadata membership distinct from the section gate', () => {
    // The metadata list drives the recents SQL exclusion + labels/icons, not
    // section membership: an unknown platform sections without being known.
    expect(MESSAGING_SESSION_SOURCE_IDS).toContain('telegram')
    expect(MESSAGING_SESSION_SOURCE_IDS).not.toContain('irc')
  })
})
