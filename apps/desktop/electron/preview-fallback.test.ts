import { describe, expect, it } from 'vitest'

import {
  homeRelativePreviewFallbackCandidates,
  isHomeRelativePreviewRef
} from './preview-fallback'

// Home-relative attachment refs stored in chat history miss every branch of
// the primary preview resolution (agent-cwd base, ~/ expansion, file: URLs),
// so the fallback must catch exactly them — and never fire for refs that
// already resolve through the primary path.
describe('isHomeRelativePreviewRef', () => {
  it('accepts home-relative refs with forward or back slashes', () => {
    expect(isHomeRelativePreviewRef('AppData/Local/hermes/attachments/foo.xlsx')).toBe(true)
    expect(isHomeRelativePreviewRef('AppData\\Local\\hermes\\attachments\\foo.xlsx')).toBe(true)
    expect(isHomeRelativePreviewRef('.hermes/attachments/foo.xlsx')).toBe(true)
  })

  it('rejects refs that resolve through the primary path', () => {
    expect(isHomeRelativePreviewRef('')).toBe(false)
    expect(isHomeRelativePreviewRef('  ')).toBe(false)
    expect(isHomeRelativePreviewRef('/home/user/report.xlsx')).toBe(false)
    expect(isHomeRelativePreviewRef('C:\\Users\\u\\AppData\\file.xlsx')).toBe(false)
    expect(isHomeRelativePreviewRef('~/attachments/foo.xlsx')).toBe(false)
    expect(isHomeRelativePreviewRef('file:///home/user/foo.xlsx')).toBe(false)
    expect(isHomeRelativePreviewRef('https://example.com/foo.xlsx')).toBe(false)
  })
})

describe('homeRelativePreviewFallbackCandidates', () => {
  const options = {
    attachmentsRoot: '/fake-home/.hermes/attachments',
    homeDir: '/fake-home'
  }

  it('joins the ref under the home dir first, deduped', () => {
    const candidates = homeRelativePreviewFallbackCandidates(
      '.hermes/attachments/foo.xlsx',
      options
    )

    // home-join wins the race: it is the real macOS/Linux layout. The
    // attachments-root join of the same ref doubles the segment and simply
    // misses on disk — harmless, but the basename retry must never add a
    // duplicate of the winner.
    expect(candidates[0]).toBe('/fake-home/.hermes/attachments/foo.xlsx')
    expect(candidates).toEqual([...new Set(candidates)])
    expect(candidates.filter((c) => c.endsWith('foo.xlsx'))).toHaveLength(candidates.length)
  })

  it('tries the basename alone under the attachments root', () => {
    const candidates = homeRelativePreviewFallbackCandidates(
      'AppData/Local/hermes/attachments/foo.xlsx',
      options
    )

    expect(candidates).toContain('/fake-home/AppData/Local/hermes/attachments/foo.xlsx')
    expect(candidates.at(-1)).toBe('/fake-home/.hermes/attachments/foo.xlsx')
  })

  it('returns no candidates for refs outside the fallback scope', () => {
    expect(homeRelativePreviewFallbackCandidates('/abs/path.xlsx', options)).toEqual([])
    expect(homeRelativePreviewFallbackCandidates('~/foo.xlsx', options)).toEqual([])
    expect(homeRelativePreviewFallbackCandidates('file:///tmp/foo.xlsx', options)).toEqual([])
    expect(homeRelativePreviewFallbackCandidates('', options)).toEqual([])
  })

  it('skips missing bases', () => {
    expect(homeRelativePreviewFallbackCandidates('foo.xlsx', {})).toEqual([])
    expect(
      homeRelativePreviewFallbackCandidates('foo.xlsx', { attachmentsRoot: '/att' })
    ).toEqual(['/att/foo.xlsx'])
  })
})
