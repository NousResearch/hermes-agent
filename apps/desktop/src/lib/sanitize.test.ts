import { describe, expect, it } from 'vitest'

import { gitRef, slug } from './sanitize'

describe('gitRef', () => {
  it('turns spaces into hyphens and keeps slashes', () => {
    expect(gitRef('beach vibes')).toBe('beach-vibes')
    expect(gitRef('feat/cool thing')).toBe('feat/cool-thing')
  })

  it('drops chars git refs forbid and collapses separators', () => {
    expect(gitRef('wip~^:?*[]')).toBe('wip')
    expect(gitRef('a   b///c..d')).toBe('a-b/c.d')
  })

  it('strips a leading separator but stays typeable (keeps a trailing one)', () => {
    expect(gitRef('/foo')).toBe('foo')
    expect(gitRef('feat/')).toBe('feat/')
  })
})

describe('slug', () => {
  it('lowercases and kebabs runs of non-alphanumerics', () => {
    expect(slug('My Profile')).toBe('my-profile')
    expect(slug('a__b  c')).toBe('a-b-c')
  })

  it('strips a leading separator but keeps a trailing one while typing', () => {
    expect(slug('--x')).toBe('x')
    expect(slug('work ')).toBe('work-')
  })
})
