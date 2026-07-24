import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { extractHermesDeepLink, parseHermesDeepLink } from './deep-link'

describe('extractHermesDeepLink', () => {
  test('finds cold-start and second-instance protocol arguments', () => {
    assert.equal(
      extractHermesDeepLink(['/opt/Hermes', '--flag', 'hermes://profile/work?new=1']),
      'hermes://profile/work?new=1'
    )
  })

  test('ignores non-array and unrelated argv input', () => {
    assert.equal(extractHermesDeepLink(null), null)
    assert.equal(extractHermesDeepLink(['/opt/Hermes', 'https://example.com']), null)
  })
})

describe('parseHermesDeepLink', () => {
  test('preserves the existing blueprint payload contract', () => {
    assert.deepEqual(parseHermesDeepLink('hermes://blueprint/morning-brief?time=08%3A00&city=New+York'), {
      kind: 'blueprint',
      name: 'morning-brief',
      params: { city: 'New York', time: '08:00' }
    })
  })

  test('accepts an explicit blank-chat request for a safe profile name', () => {
    assert.deepEqual(parseHermesDeepLink('hermes://profile/research_bot?new=1'), {
      kind: 'profile',
      name: 'research_bot',
      params: { new: '1' }
    })
  })

  test('accepts percent-encoding without weakening profile validation', () => {
    assert.deepEqual(parseHermesDeepLink('hermes://profile/%77ork?new=1'), {
      kind: 'profile',
      name: 'work',
      params: { new: '1' }
    })
    assert.equal(parseHermesDeepLink('hermes://profile/%2e%2e%2fwork?new=1'), null)
  })

  test('rejects profile links that carry another action or prompt-shaped data', () => {
    assert.equal(parseHermesDeepLink('hermes://profile/work'), null)
    assert.equal(parseHermesDeepLink('hermes://profile/work?new=0'), null)
    assert.equal(parseHermesDeepLink('hermes://profile/work?new=1&prompt=hello'), null)
    assert.equal(parseHermesDeepLink('hermes://profile/work?new=1&new=1'), null)
  })

  test('rejects unsafe names, extra path segments, fragments, and unsupported routes', () => {
    assert.equal(parseHermesDeepLink('hermes://profile/Work?new=1'), null)
    assert.equal(parseHermesDeepLink('hermes://profile/work/chat?new=1'), null)
    assert.equal(parseHermesDeepLink('hermes://profile/work?new=1#prompt'), null)
    assert.equal(parseHermesDeepLink('hermes://unknown/work?new=1'), null)
    assert.equal(parseHermesDeepLink('not a url'), null)
  })
})
