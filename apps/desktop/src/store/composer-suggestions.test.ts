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
})
