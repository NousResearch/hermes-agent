import { describe, expect, it } from 'vitest'

import { extractLinkPreviewTargets } from './content'

describe('extractLinkPreviewTargets', () => {
  it('returns empty for empty text', () => {
    expect(extractLinkPreviewTargets('')).toEqual([])
    expect(extractLinkPreviewTargets('no links here')).toEqual([])
  })

  it('extracts http(s) URLs in order, deduped', () => {
    const text = 'see https://a.dev/x and https://b.dev then https://a.dev/x again'
    expect(extractLinkPreviewTargets(text)).toEqual(['https://a.dev/x', 'https://b.dev'])
  })

  it('excludes localhost-shaped hosts', () => {
    const text = 'local dev at http://localhost:3000 and https://127.0.0.1:8080 and https://[::1]:5173 plus https://real.dev'
    expect(extractLinkPreviewTargets(text)).toEqual(['https://real.dev'])
  })

  it('drops trailing punctuation from prose-embedded URLs', () => {
    expect(extractLinkPreviewTargets('read https://a.dev/docs.')).toEqual(['https://a.dev/docs'])
    expect(extractLinkPreviewTargets('read https://a.dev/docs, then more')).toEqual(['https://a.dev/docs'])
  })

  it('skips non-URL garbage that merely contains a colon-slash', () => {
    expect(extractLinkPreviewTargets('https:// is empty host')).toEqual([])
  })

  it('excludes private-looking but syntactically valid hosts at extraction level only for localhost shapes', () => {
    // Deeper private-range decisions belong to the main-process resolver;
    // extraction only filters what can never be public per hostname.
    expect(extractLinkPreviewTargets('https://192.168.1.10/x')).toEqual(['https://192.168.1.10/x'])
  })
})
