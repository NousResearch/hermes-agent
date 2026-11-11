import { describe, expect, it } from 'vitest'

import { resolveBrandIcon } from './brand-icon'

describe('resolveBrandIcon', () => {
  it('resolves a registrable domain to a brand glyph', () => {
    expect(resolveBrandIcon('github.com')).toBeTruthy()
    expect(resolveBrandIcon('gitlab.com')).toBeTruthy()
  })

  it('ignores case and a leading www.', () => {
    const github = resolveBrandIcon('github.com')

    expect(resolveBrandIcon('WWW.GitHub.com')).toBe(github)
  })

  it('inherits the parent brand on subdomains', () => {
    const github = resolveBrandIcon('github.com')

    expect(resolveBrandIcon('gist.github.com')).toBe(github)
    expect(resolveBrandIcon('api.github.com')).toBe(github)
  })

  it('prefers the longest matching suffix over its parent', () => {
    const google = resolveBrandIcon('google.com')
    const docs = resolveBrandIcon('docs.google.com')

    expect(docs).toBeTruthy()
    expect(docs).not.toBe(google)
  })

  it('returns null for unknown hosts and bare labels', () => {
    expect(resolveBrandIcon('example.com')).toBeNull()
    expect(resolveBrandIcon('localhost')).toBeNull()
    expect(resolveBrandIcon('')).toBeNull()
  })

  it('never matches on a bare public suffix', () => {
    // `github.io` is a real entry; a lone `io` must not inherit it.
    expect(resolveBrandIcon('io')).toBeNull()
    expect(resolveBrandIcon('com')).toBeNull()
  })
})
