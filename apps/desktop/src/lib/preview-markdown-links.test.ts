import { describe, expect, it } from 'vitest'

import {
  classifyPreviewMarkdownHref,
  githubHeadingSlug,
  nextHeadingId,
  resolvePreviewFileHref,
  rewritePreviewFileLinks
} from './preview-markdown-links'

describe('preview markdown link classification', () => {
  const note = '/vault/notes/index.md'

  it('sends https and mailto through the external opener, not a file path', () => {
    expect(classifyPreviewMarkdownHref('https://example.com/docs', note)).toEqual({
      kind: 'external',
      href: 'https://example.com/docs'
    })
    expect(classifyPreviewMarkdownHref('mailto:a@example.com', note)).toEqual({
      kind: 'external',
      href: 'mailto:a@example.com'
    })
    expect(classifyPreviewMarkdownHref('//example.com/docs', note)).toEqual({
      kind: 'external',
      href: 'https://example.com/docs'
    })
  })

  it('keeps a heading fragment inside the note', () => {
    expect(classifyPreviewMarkdownHref('#managed-tiered-kv-cache', note)).toEqual({
      kind: 'hash',
      fragment: 'managed-tiered-kv-cache'
    })
    expect(classifyPreviewMarkdownHref('#Hello%20World', note)).toEqual({
      kind: 'hash',
      fragment: 'Hello World'
    })
  })

  it('resolves a sibling note against the file being previewed', () => {
    expect(classifyPreviewMarkdownHref('../other.md', note)).toEqual({
      kind: 'file',
      path: '/vault/other.md'
    })
    expect(classifyPreviewMarkdownHref('./nested/a.md', note)).toEqual({
      kind: 'file',
      path: '/vault/notes/nested/a.md'
    })
    expect(resolvePreviewFileHref('file:///vault/other.md', note)).toBe('/vault/other.md')
  })

  it('does not turn a markdown filename into a website', () => {
    expect(classifyPreviewMarkdownHref('notes.md', note)?.kind).toBe('file')
    expect(classifyPreviewMarkdownHref('notes.md')?.kind).toBe('inert')
  })

  it('drops scriptable schemes instead of opening them', () => {
    expect(classifyPreviewMarkdownHref('javascript:alert(1)', note)).toEqual({ kind: 'inert' })
    expect(classifyPreviewMarkdownHref('data:text/html,hi', note)).toEqual({ kind: 'inert' })
  })

  it('joins a windows relative link without leaving the drive', () => {
    expect(resolvePreviewFileHref('..\\other.md', 'C:\\vault\\notes\\index.md')).toBe('C:/vault/other.md')
  })

  it('rewrites a relative link to a file url before the hardener can collapse it', () => {
    const tree = {
      type: 'root',
      children: [{ type: 'link', url: '../other.md', children: [{ type: 'text' }] }]
    }

    rewritePreviewFileLinks(tree, note)

    expect(tree.children[0]?.url).toBe('https://preview-file.invalid/vault/other.md')
    expect(classifyPreviewMarkdownHref(tree.children[0]?.url, note)).toEqual({
      kind: 'file',
      path: '/vault/other.md'
    })
    expect(classifyPreviewMarkdownHref('https://example.com', note)).toEqual({
      kind: 'external',
      href: 'https://example.com'
    })
  })

  it('slugs headings the way a generated table of contents addresses them', () => {
    expect(githubHeadingSlug('Managed Tiered KV Cache')).toBe('managed-tiered-kv-cache')
    expect(githubHeadingSlug('Hello, World!')).toBe('hello-world')

    const counts = new Map<string, number>()

    expect(nextHeadingId('Intro', counts)).toBe('intro')
    expect(nextHeadingId('Intro', counts)).toBe('intro-1')
  })
})
