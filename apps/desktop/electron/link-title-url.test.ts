import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { canonicalTitleCacheKey, isFetchableHttpUrl } from './link-title-url'

// #93893: the renderer can send ANY href-shaped string to the
// hermes:fetchLinkTitle IPC; before these guards, an unparseable string
// became its own cache key (canonicalTitleCacheKey returned the raw value)
// and flowed to the hidden title window's loadURL(), producing repeating
// `Failed to load URL: … ERR_NAME_NOT_RESOLVED` console noise.

describe('isFetchableHttpUrl', () => {
  test('admits absolute http and https URLs', () => {
    assert.equal(isFetchableHttpUrl('https://example.com/docs'), true)
    assert.equal(isFetchableHttpUrl('http://example.com'), true)
    assert.equal(isFetchableHttpUrl('https://example.com/a/b?x=1'), true)
  })

  test('rejects leaked directive markup before it can reach loadURL', () => {
    assert.equal(isFetchableHttpUrl('@url:`https://oauth2:%s@example.internal.host`'), false)
    assert.equal(isFetchableHttpUrl('@url:https://example.com'), false)
  })

  test('rejects non-http schemes, placeholders, and garbage', () => {
    assert.equal(isFetchableHttpUrl('file:///etc/passwd'), false)
    assert.equal(isFetchableHttpUrl('mailto:user@example.com'), false)
    assert.equal(isFetchableHttpUrl('javascript:alert(1)'), false)
    assert.equal(isFetchableHttpUrl('https://example.com'), true) // control
    assert.equal(isFetchableHttpUrl('not a url'), false)
    assert.equal(isFetchableHttpUrl('printf("hello %s")'), false)
    assert.equal(isFetchableHttpUrl(''), false)
  })
})

describe('canonicalTitleCacheKey', () => {
  test('never returns the raw string for unparseable input', () => {
    // The passthrough hole: an unparseable value used to become a
    // loadable-looking cache key. It must collapse to '' instead.
    const junk = '@url:`https://oauth2:%s@example.internal.host`'
    assert.equal(canonicalTitleCacheKey(junk), '')
    assert.equal(canonicalTitleCacheKey('not a url'), '')
    assert.equal(canonicalTitleCacheKey(''), '')
  })

  test('builds a host+path+search key, normalizing www and trailing slashes', () => {
    assert.equal(canonicalTitleCacheKey('https://www.example.com/docs/'), 'example.com/docs')
    assert.equal(canonicalTitleCacheKey('https://example.com'), 'example.com/')
    assert.equal(canonicalTitleCacheKey('https://example.com/search?q=hi'), 'example.com/search?q=hi')
    // Same page, same key (that is the point of a cache key).
    assert.equal(
      canonicalTitleCacheKey('http://www.example.com/docs///'),
      canonicalTitleCacheKey('https://example.com/docs')
    )
  })
})
