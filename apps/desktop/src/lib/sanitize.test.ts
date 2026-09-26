import { describe, expect, it } from 'vitest'

import { gitRef, projectName, slug } from './sanitize'

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

describe('projectName', () => {
  it('drops zero-width/format characters that silently break name matching', () => {
    expect(projectName('AttendanceSync\u200cBot')).toBe('AttendanceSyncBot') // ZWNJ
    expect(projectName('a\u200bb')).toBe('ab') // ZWSP
    expect(projectName('a\u200db')).toBe('ab') // ZWJ — splits emoji sequences too, by design
    expect(projectName('a\ufeffb')).toBe('ab') // BOM
    expect(projectName('a\u00adb')).toBe('ab') // soft hyphen
    expect(projectName('a\u202eb')).toBe('ab') // RTL override
  })

  it('drops control characters and the trailing specials block', () => {
    expect(projectName('a\u0000b')).toBe('ab')
    expect(projectName('a\u0007b')).toBe('ab')
    expect(projectName('a\u007fb')).toBe('ab')
    expect(projectName('a\u2028b')).toBe('ab')
    expect(projectName('a\ufffbb')).toBe('ab')
  })

  it('keeps visible text — ASCII, CJK, spaces, emoji — intact', () => {
    expect(projectName('My 项目 🚀')).toBe('My 项目 🚀')
    expect(projectName(' plain spaced ')).toBe(' plain spaced ')
  })

  it('empties an invisible-only name so submit stays disabled', () => {
    expect(projectName('\u200b\u200c\u200d\ufeff')).toBe('')
  })
})
