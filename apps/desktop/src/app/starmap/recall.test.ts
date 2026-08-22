import { describe, expect, it } from 'vitest'

import { resolveRecallTarget } from './recall'

// Regression: multi-profile nodes arrive with PREFIXED ids (`<profile>:<id>`)
// while the server only knows original ids — recall/insert must unprefix and
// scope to the node's OWN profile or every non-active-profile node 404s
// ("Could not load that memory to insert." toast, 2026-08-22).
describe('resolveRecallTarget', () => {
  it('passes single-profile nodes through untouched (colons in id are NOT a prefix)', () => {
    expect(resolveRecallTarget({ id: 'memory:honcho:1197', kind: 'memory', label: 'x' }))
      .toEqual({ id: 'memory:honcho:1197' })
    expect(resolveRecallTarget({ id: 'my-skill', kind: 'skill', label: 'x' }))
      .toEqual({ id: 'my-skill' })
  })

  it('prefers the explicit _originalId and scopes to the node profile', () => {
    expect(resolveRecallTarget({
      _originalId: 'memory:honcho:1197',
      id: 'default:memory:honcho:1197',
      kind: 'memory',
      label: 'x',
      profile: 'default'
    })).toEqual({ id: 'memory:honcho:1197', profile: 'default' })
  })

  it('falls back to stripping a verified profile prefix', () => {
    expect(resolveRecallTarget({
      id: 'test-deconfliction-bot:memory:memory:3',
      kind: 'memory',
      label: 'x',
      profile: 'test-deconfliction-bot'
    })).toEqual({ id: 'memory:memory:3', profile: 'test-deconfliction-bot' })
  })

  it('never strips when the id does not actually start with the profile prefix', () => {
    // e.g. a raw id whose first segment coincidentally matches nothing — the
    // strip must be verified, not a blind replace.
    expect(resolveRecallTarget({
      id: 'memory:default:2',
      kind: 'memory',
      label: 'x',
      profile: 'default'
    })).toEqual({ id: 'memory:default:2', profile: 'default' })
  })
})
