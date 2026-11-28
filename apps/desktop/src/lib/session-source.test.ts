import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'

import { isMessagingSource, sessionSourceLabel, sessionSourceSearchTerms } from './session-source'

// Regression guard for #46761 / PR #47395: Photon (iMessage) must keep its own
// sidebar section. refreshMessagingSessions() filters rows through
// isMessagingSource(), so this entry is the sole condition that keeps Photon
// sessions out of generic recents. A silent removal would regress the feature
// with no test failure — these asserts pin the contract.
describe('photon messaging source registration', () => {
  it('treats photon as a messaging source (own sidebar section)', () => {
    expect(isMessagingSource('photon')).toBe(true)
  })

  it('is case/space insensitive on the source id', () => {
    expect(isMessagingSource('PHOTON')).toBe(true)
    expect(isMessagingSource('  photon ')).toBe(true)
  })

  it('exposes the iMessage/messages search aliases so Photon sessions are findable', () => {
    const terms = sessionSourceSearchTerms('photon')
    expect(terms).toContain('imessage')
    expect(terms).toContain('messages')
  })

  it('does not flag local/CLI-ish sources as messaging (guard sanity)', () => {
    expect(isMessagingSource('cli')).toBe(false)
    expect(isMessagingSource(null)).toBe(false)
    expect(isMessagingSource(undefined)).toBe(false)
  })
})

it('applies profile-scoped source labels live and restores defaults when the plugin unloads', () => {
  const area = 'session.sourceLabels'
  const session = { id: 'thread', profile: 'default' }
  const original = sessionSourceLabel('photon')

  const { result, unmount } = renderHook(() => {
    const contributions = useContributions(area)

    return { contributions, label: sessionSourceLabel('photon', session) }
  })

  let dispose = () => {}

  try {
    act(() => {
      dispose = registry.register({
        id: 'test-channel-labels:messages',
        source: 'plugin:test-channel-labels',
        area,
        data: { source: 'PHOTON', label: 'iMessage', profile: 'default' }
      })
    })

    expect(result.current.label).toBe('iMessage')
    expect(sessionSourceLabel('photon', { ...session, profile: 'work' })).toBe(original)
    expect(sessionSourceLabel('photon')).toBe(original)
    expect(sessionSourceLabel('api_server', session)).toBe('API')

    act(() => dispose())

    expect(result.current.label).toBe(original)
    expect(result.current.contributions).toEqual([])
  } finally {
    act(() => dispose())
    unmount()
  }
})
