/**
 * Roster payload fingerprints: the content-level change detection that keeps
 * the 5s roster poll from republishing identical payloads (new object graph
 * → new array reference → every shared-atom subscriber re-renders, and on
 * macOS the pane churn drops IME input-method focus).
 *
 * Pinned behaviors:
 *  - same content, different references/order ⇒ same fingerprint;
 *  - any observable field (labels, session activity, source status, avatar
 *    flag, server ui_meta) ⇒ different fingerprint;
 *  - local-only payload noise (ui_meta image data URLs, revisions bookkeeping,
 *    ghost twins) never changes the fingerprint.
 */

import { describe, expect, it } from 'vitest'

import { rosterPayloadFingerprint } from './roster-fingerprint'
import type { RosterRow } from './types'

/** A realistic rich row as profiles.list returns it. */
function bot(overrides: Partial<RosterRow> = {}): RosterRow {
  return {
    name: 'scribe',
    display_name: 'Scribe',
    title: 'Scribe',
    has_avatar: true,
    connectionId: 'local',
    connectionKind: 'local',
    connectionLabel: 'Local',
    remoteSource: false,
    sourceReachable: true,
    canonical_session: {
      id: 'sess-1',
      resolved_id: 'sess-1',
      last_active: 1_753_000_000,
      preview: 'hello',
      title: 'Bot Chat'
    },
    last_session: {
      last_active: 1_752_900_000,
      preview: 'older work'
    },
    worker_session: { last_active: 1_752_950_000 },
    ui_meta: {
      'hermes-bots': {
        sectionId: 'writing',
        color: '#c0ffee',
        groups: ['docs'],
        pinned: true,
        title: 'Scribe'
      }
    },
    ...overrides
  }
}

describe('rosterPayloadFingerprint', () => {
  it('ignores object identity: same content, fresh references', () => {
    const a = rosterPayloadFingerprint([bot()])
    const b = rosterPayloadFingerprint([bot()])

    expect(a).toBe(b)
  })

  it('ignores wire order and the pane pin/activity sort', () => {
    const base = [bot(), bot({ name: 'researcher', display_name: 'Researcher' })]
    const shuffled = [bot({ name: 'researcher', display_name: 'Researcher' }), bot()]

    expect(rosterPayloadFingerprint(base)).toBe(rosterPayloadFingerprint(shuffled))
  })

  it('distinguishes same-named rows across connections', () => {
    const local = rosterPayloadFingerprint([bot({ connectionId: 'local' })])
    const remote = rosterPayloadFingerprint([bot({ connectionId: 'ssh::box' })])

    expect(local).not.toBe(remote)
  })

  it('ignores offline-owner ghost twins', () => {
    const plain = rosterPayloadFingerprint([bot()])
    const withGhost = rosterPayloadFingerprint([
      bot(),
      { ghost: true, name: 'scribe', connectionId: 'local', remoteSource: false }
    ])

    expect(plain).toBe(withGhost)
  })

  it('detects identity and label changes', () => {
    const base = rosterPayloadFingerprint([bot()])

    for (const changed of [
      bot({ name: 'scribe2' }),
      bot({ display_name: 'Scribe Prime' }),
      bot({ title: 'Prime' }),
      bot({ handle: '@scribe' }),
      bot({ description: 'writes' })
    ]) {
      expect(rosterPayloadFingerprint([changed])).not.toBe(base)
    }
  })

  it('detects session-activity changes (age, unread watermark, sort)', () => {
    const base = rosterPayloadFingerprint([bot()])

    expect(
      rosterPayloadFingerprint([bot({ canonical_session: { id: 'sess-1', last_active: 1_753_100_000 } })])
    ).not.toBe(base)
    expect(rosterPayloadFingerprint([bot({ last_session: { last_active: 1_753_000_001 } })])).not.toBe(base)
    expect(rosterPayloadFingerprint([bot({ worker_session: { last_active: 1_753_000_002 } })])).not.toBe(base)
  })

  it('detects source/status and avatar-flag changes', () => {
    const base = rosterPayloadFingerprint([bot()])

    for (const changed of [
      bot({ sourceReachable: false }),
      bot({ sourceError: 'ssh: refused' }),
      bot({ remoteSource: true, connectionId: 'ssh::box', connectionKind: 'ssh' }),
      bot({ has_avatar: false }),
      bot({ targetProfile: 'other' })
    ]) {
      expect(rosterPayloadFingerprint([changed])).not.toBe(base)
    }
  })

  it('detects server ui_meta changes (sections, pins, hidden, meta title)', () => {
    const base = rosterPayloadFingerprint([bot()])

    for (const ui of [
      { 'hermes-bots': { sectionId: 'research', pinned: true, title: 'Scribe' } },
      { 'hermes-bots': { sectionId: 'writing', pinned: false, title: 'Scribe' } },
      { 'hermes-bots': { sectionId: 'writing', pinned: true, title: 'Scribe', hidden: true } }
    ]) {
      expect(rosterPayloadFingerprint([bot({ ui_meta: ui })])).not.toBe(base)
    }
  })

  it('ignores local-only payload noise: image data URLs and revisions', () => {
    const base = rosterPayloadFingerprint([bot()])

    const withImage = {
      ...bot(),
      ui_meta: {
        'hermes-bots': { ...bot().ui_meta!['hermes-bots']!, image: `data:image/png;base64,${'A'.repeat(4096)}` }
      }
    }

    const withRevisions = bot({ ui_meta_revisions: { 'hermes-bots': 7 } })

    expect(rosterPayloadFingerprint([withImage])).toBe(base)
    expect(rosterPayloadFingerprint([withRevisions])).toBe(base)
  })

  it('ignores foreign ui_meta keys', () => {
    const hermesBots = bot().ui_meta!['hermes-bots']

    const base = rosterPayloadFingerprint([bot()])
    const foreign = rosterPayloadFingerprint([
      bot({ ui_meta: { 'hermes-bots': hermesBots, 'other-plugin': { thing: 1 } } })
    ])

    expect(foreign).toBe(base)
  })

  it('treats missing hermes-bots meta as its own content', () => {
    const base = rosterPayloadFingerprint([bot()])
    const bare = rosterPayloadFingerprint([{ ...bot(), ui_meta: undefined }])
    const foreignOnly = rosterPayloadFingerprint([bot({ ui_meta: { 'other-plugin': { thing: 1 } } })])

    // No hermes-bots meta ⇒ no server overlay to apply — distinct content.
    expect(bare).not.toBe(base)
    expect(foreignOnly).toBe(bare)
  })

  it('treats an empty roster as stable content', () => {
    expect(rosterPayloadFingerprint([])).toBe(rosterPayloadFingerprint([]))
    expect(rosterPayloadFingerprint([])).not.toBe(rosterPayloadFingerprint([bot()]))
  })
})