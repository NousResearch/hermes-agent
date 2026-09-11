import { describe, expect, it } from 'vitest'

import {
  $composerSuggestionsBySession,
  type ComposerSuggestion,
  markSuggestionInvoked,
  offerSuggestions,
  suggestionKey
} from './composer-suggestions'

const suggestion = (id: string, provider = 'test'): ComposerSuggestion => ({
  doneLabel: 'done',
  doneTip: 'done',
  id,
  invoke: async () => {},
  label: id,
  provider,
  tip: 'because',
  workingLabel: 'working',
  workingTip: 'working'
})

const pillsFor = (sessionId: string) => ($composerSuggestionsBySession.get()[sessionId] ?? []).map(s => s.id)

describe('composer suggestion bus', () => {
  it('publishes event offerings per session and withdraws on empty offer', () => {
    offerSuggestions('s1', 'test', [suggestion('a')])

    expect(pillsFor('s1')).toEqual(['a'])
    expect(pillsFor('s2')).toEqual([])

    offerSuggestions('s1', 'test', [])

    expect(pillsFor('s1')).toEqual([])
  })

  it('caps merged suggestions at two', () => {
    offerSuggestions('s3', 'test', [suggestion('a'), suggestion('b'), suggestion('c')])

    expect(pillsFor('s3')).toHaveLength(2)

    offerSuggestions('s3', 'test', [])
  })

  it('dedupes by provider-namespaced key across providers', () => {
    offerSuggestions('s4', 'p1', [suggestion('same', 'p1')])
    offerSuggestions('s4', 'p2', [suggestion('same', 'p2')])

    // Different providers, same id — distinct keys, both allowed.
    expect(pillsFor('s4')).toEqual(['same', 'same'])

    offerSuggestions('s4', 'p1', [])
    offerSuggestions('s4', 'p2', [])
  })

  it('quiets a suggestion after it is repeatedly withdrawn uninvoked', () => {
    // Three offer/withdraw cycles = three strikes.
    for (let i = 0; i < 3; i += 1) {
      offerSuggestions('s5', 'test', [suggestion('naggy')])
      offerSuggestions('s5', 'test', [])
    }

    offerSuggestions('s5', 'test', [suggestion('naggy')])

    expect(pillsFor('s5')).toEqual([])

    offerSuggestions('s5', 'test', [])
  })

  it('an invoked suggestion never accrues strikes', () => {
    for (let i = 0; i < 3; i += 1) {
      offerSuggestions('s6', 'test', [suggestion('used')])
      markSuggestionInvoked('s6', suggestionKey(suggestion('used')))
      offerSuggestions('s6', 'test', [])
    }

    offerSuggestions('s6', 'test', [suggestion('used')])

    expect(pillsFor('s6')).toEqual(['used'])

    offerSuggestions('s6', 'test', [])
  })

  it('replaces an offer whose rendered copy changed under the same key', () => {
    offerSuggestions('s7', 'test', [{ ...suggestion('linear'), tip: 'because you mentioned “linear”' }])
    offerSuggestions('s7', 'test', [{ ...suggestion('linear'), tip: 'because you pasted linear.app' }])

    // Same key, new trigger — the strip must paint the new reason, not the
    // first one it ever saw.
    expect(($composerSuggestionsBySession.get().s7 ?? []).map(s => s.tip)).toEqual(['because you pasted linear.app'])

    offerSuggestions('s7', 'test', [])
  })

  it('re-offering the same key swaps in the fresh invoke closure', async () => {
    const calls: string[] = []

    const offer = (tag: string) =>
      offerSuggestions('s8', 'test', [
        { ...suggestion('linear'), invoke: async () => void calls.push(tag), label: `Add linear ${tag}` }
      ])

    offer('first')
    offer('second')

    await ($composerSuggestionsBySession.get().s8 ?? [])[0]!.invoke({ cancelled: () => false, sessionId: 's8' })

    // A pinned first object means the pill runs work built for a draft the
    // user has since changed.
    expect(calls).toEqual(['second'])

    offerSuggestions('s8', 'test', [])
  })

  it('keeps the array reference when nothing the pill paints changed', () => {
    offerSuggestions('s9', 'test', [suggestion('linear')])

    const first = $composerSuggestionsBySession.get().s9

    offerSuggestions('s9', 'test', [suggestion('linear')])

    expect($composerSuggestionsBySession.get().s9).toBe(first)

    offerSuggestions('s9', 'test', [])
  })
})
