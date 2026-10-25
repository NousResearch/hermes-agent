// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import {
  findPreviewHeading,
  isPreviewFileHref,
  localImageSrcPath,
  noteDirectory,
  previewHeadingSlug,
  rehypePreviewFileLinks,
  rehypePreviewHeadingIds,
  rehypePreviewImagePaths,
  remarkPreviewFileLinks,
  rewriteLocalImageSrc
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

describe('rewriteLocalImageSrc', () => {
  it('resolves note-relative paths against the note directory', () => {
    expect(rewriteLocalImageSrc('docs/img.png', '/work/notes')).toBe('/work/notes/docs/img.png')
    expect(rewriteLocalImageSrc('./img.png', '/work/notes')).toBe('/work/notes/img.png')
    expect(rewriteLocalImageSrc('../shared/img.png', '/work/notes')).toBe('/work/shared/img.png')
    expect(rewriteLocalImageSrc('/abs/img.png', '/work/notes')).toBe('/abs/img.png')
    expect(rewriteLocalImageSrc('docs/my shot.png', '/work/notes')).toBe('/work/notes/docs/my%20shot.png')
    expect(rewriteLocalImageSrc('docs/my%20shot.png', '/work/notes')).toBe('/work/notes/docs/my%20shot.png')
  })

  it('leaves web and inline sources for the hardener to handle', () => {
    expect(rewriteLocalImageSrc('https://example.com/x.png', '/work/notes')).toBeNull()
    expect(rewriteLocalImageSrc('data:image/png;base64,AAAA', '/work/notes')).toBeNull()
    expect(rewriteLocalImageSrc('~/shots/img.png', '/work/notes')).toBeNull()
    expect(rewriteLocalImageSrc('docs/img.png')).toBeNull()
  })

  it('marks Windows paths so the hardener keeps them', () => {
    expect(rewriteLocalImageSrc('C:\\shots\\a.png', 'C:\\vault')).toBe('/C:/shots/a.png')
    expect(rewriteLocalImageSrc('file:///C:/shots/a.png', undefined)).toBe('/C:/shots/a.png')
    expect(rewriteLocalImageSrc('file:///Users/mac/a.png', undefined)).toBe('/Users/mac/a.png')
  })
})

describe('localImageSrcPath', () => {
  it('restores the readable path behind a rewritten src', () => {
    expect(localImageSrcPath('/work/notes/docs/img.png')).toBe('/work/notes/docs/img.png')
    expect(localImageSrcPath('/work/notes/docs/my%20shot.png')).toBe('/work/notes/docs/my shot.png')
    expect(localImageSrcPath('/C:/shots/a.png')).toBe('C:/shots/a.png')
  })

  it('returns null for sources an img element loads directly', () => {
    expect(localImageSrcPath('https://example.com/x.png')).toBeNull()
    expect(localImageSrcPath('data:image/png;base64,AAAA')).toBeNull()
    expect(localImageSrcPath('')).toBeNull()
  })
})

describe('rehypePreviewImagePaths', () => {
  it('rewrites img srcs and touches nothing else', () => {
    const tree = {
      children: [
        {
          children: [
            { properties: { src: 'docs/img.png' }, tagName: 'img', type: 'element' },
            { properties: { href: 'docs/other.md' }, tagName: 'a', type: 'element' }
          ],
          tagName: 'p',
          type: 'element'
        }
      ],
      type: 'root'
    }

    rehypePreviewImagePaths('/vault/notes')(tree)

    const [paragraph] = tree.children
    const [img, anchor] = paragraph.children

    expect(img.properties).toEqual({ src: '/vault/notes/docs/img.png' })
    expect(anchor.properties).toEqual({ href: 'docs/other.md' })
  })
})

describe('rehypePreviewFileLinks', () => {
  const anchor = (href: string) => ({ children: [], properties: { href }, tagName: 'a', type: 'element' })

  it('rewrites raw-HTML file anchors to the preview door and leaves web and scheme links alone', () => {
    const img = { children: [], properties: { src: 'docs/img.png' }, tagName: 'img', type: 'element' }

    const tree = {
      children: [
        {
          children: [
            anchor('README.md'),
            anchor('https://example.com'),
            anchor('#亮点'),
            anchor('#a%20b'),
            anchor('mailto:a@b.c'),
            img
          ],
          tagName: 'p',
          type: 'element'
        }
      ],
      type: 'root'
    }

    rehypePreviewFileLinks()(tree)

    const [paragraph] = tree.children
    const nodes = paragraph.children

    expect(nodes[0].properties).toEqual({ href: `#preview/${encodeURIComponent('README.md')}` })
    expect(nodes[1].properties).toEqual({ href: 'https://example.com' })
    // A raw CJK fragment is percent-encoded so the hardener keeps it;
    // an already-encoded one survives untouched (no double-encoding).
    expect(nodes[2].properties).toEqual({ href: `#${encodeURIComponent('亮点')}` })
    expect(nodes[3].properties).toEqual({ href: '#a%20b' })
    expect(nodes[4].properties).toEqual({ href: 'mailto:a@b.c' })
    expect(nodes[5].properties).toEqual({ src: 'docs/img.png' })
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
