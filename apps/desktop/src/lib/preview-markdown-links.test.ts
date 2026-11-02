// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import {
  findPreviewHeading,
  isPreviewFileHref,
  noteDirectory,
  previewHeadingSlug,
  rehypePreviewHeadingIds,
  remarkPreviewFileLinks
} from './preview-markdown-links'

describe('previewHeadingSlug', () => {
  it('matches the GitHub slug a generated table of contents links to', () => {
    expect(previewHeadingSlug('Managed Tiered KV Cache')).toBe('managed-tiered-kv-cache')
    expect(previewHeadingSlug('  Hello, World!  ')).toBe('hello-world')
    expect(previewHeadingSlug('a -- b')).toBe('a-b')
  })

  it('slugs composed and decomposed spellings alike', () => {
    const composed = '\u30D3'
    const decomposed = '\u30D2\u3099'

    expect(previewHeadingSlug(decomposed)).toBe(previewHeadingSlug(composed))
  })
})

describe('rehypePreviewHeadingIds', () => {
  const heading = (tagName: string, text: string) => ({
    children: [{ type: 'text', value: text }],
    properties: {},
    tagName,
    type: 'element'
  })

  it('stamps ids and numbers duplicates', () => {
    const tree = {
      children: [heading('h2', 'Setup'), heading('h3', 'Setup'), heading('h2', '!!!'), heading('p', 'Setup')],
      type: 'root'
    }

    rehypePreviewHeadingIds()(tree)

    const [first, second, empty, paragraph] = tree.children

    expect(first.properties).toEqual({ id: 'setup' })
    expect(second.properties).toEqual({ id: 'setup-1' })
    expect(empty.properties).toEqual({})
    expect(paragraph.properties).toEqual({})
  })
})

describe('remarkPreviewFileLinks', () => {
  const link = (url: string) => ({ children: [], type: 'link', url })

  it('wraps file links in the preview hash and leaves web and anchor links alone', () => {
    const tree = {
      children: [
        link('../other.md'),
        link('/abs/note.md'),
        link('file:///tmp/a.md'),
        link('https://example.com'),
        link('mailto:a@b.c'),
        link('#heading'),
        link('www.example.com'),
        link('//cdn.example.com/x')
      ],
      type: 'root'
    }

    remarkPreviewFileLinks()(tree)

    expect(tree.children.map(node => node.url)).toEqual([
      `#preview/${encodeURIComponent('../other.md')}`,
      `#preview/${encodeURIComponent('/abs/note.md')}`,
      `#preview/${encodeURIComponent('file:///tmp/a.md')}`,
      'https://example.com',
      'mailto:a@b.c',
      '#heading',
      'www.example.com',
      '//cdn.example.com/x'
    ])
  })

  it('classifies hrefs', () => {
    expect(isPreviewFileHref('notes/a.md')).toBe(true)
    expect(isPreviewFileHref('C:\\notes\\a.md')).toBe(true)
    expect(isPreviewFileHref('javascript:alert(1)')).toBe(false)
    expect(isPreviewFileHref('')).toBe(false)
  })
})

describe('noteDirectory', () => {
  it('returns the directory the note lives in', () => {
    expect(noteDirectory('/vault/notes/index.md')).toBe('/vault/notes')
    expect(noteDirectory('/index.md')).toBe('/')
    expect(noteDirectory('C:\\vault\\index.md')).toBe('C:\\vault')
    expect(noteDirectory('index.md')).toBeUndefined()
    expect(noteDirectory(undefined)).toBeUndefined()
  })
})

describe('findPreviewHeading', () => {
  it('finds a heading by id, by re-slugged text, and across Unicode normalization', () => {
    const root = document.createElement('div')

    root.innerHTML = `<h2 id="setup">Setup</h2><h2 id="${previewHeadingSlug('\u30D3')}">ビ</h2>`

    expect(findPreviewHeading(root, 'setup')?.textContent).toBe('Setup')
    expect(findPreviewHeading(root, 'Setup')?.textContent).toBe('Setup')
    expect(findPreviewHeading(root, encodeURIComponent('\u30D2\u3099'))?.textContent).toBe('ビ')
    expect(findPreviewHeading(root, 'missing')).toBeNull()
  })
})
