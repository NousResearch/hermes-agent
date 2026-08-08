import { describe, expect, it } from 'vitest'

import {
  findHeadingByHash,
  githubSlug,
  hashFragment,
  rehypeHeadingIds
} from './markdown-heading-ids'

interface HastNode {
  children?: HastNode[]
  properties?: Record<string, unknown>
  tagName?: string
  type?: string
  value?: string
}

function heading(tag: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6', text: string): HastNode {
  return {
    type: 'element',
    tagName: tag,
    properties: {},
    children: [{ type: 'text', value: text }],
  }
}

function wrap(...children: HastNode[]): HastNode {
  return { type: 'root', children }
}

describe('githubSlug', () => {
  it('lowercases and joins words with single dashes', () => {
    expect(githubSlug('Managed Tiered KV Cache')).toBe('managed-tiered-kv-cache')
  })

  it('strips punctuation but keeps unicode letters and digits', () => {
    expect(githubSlug('Hello, world! 123')).toBe('hello-world-123')
  })

  it('keeps combining marks and dashes', () => {
    expect(githubSlug('café — naïve')).toContain('caf')
    expect(githubSlug('café — naïve')).toContain('na')
  })
})

describe('hashFragment', () => {
  it('strips the leading hash', () => {
    expect(hashFragment('#foo')).toBe('foo')
  })

  it('decodes percent-escapes', () => {
    expect(hashFragment('#hello%20world')).toBe('hello world')
  })

  it('returns the raw value when decoding fails', () => {
    // %ZZ is not a valid escape; decodeURIComponent throws — the helper
    // returns the raw fragment so a malformed href doesn't crash the
    // preview.
    expect(hashFragment('#bad%ZZescape')).toBe('bad%ZZescape')
  })

  it('returns empty string for "#"', () => {
    expect(hashFragment('#')).toBe('')
  })
})

describe('rehypeHeadingIds', () => {
  it('stamps ids matching github-slugger', () => {
    const tree = wrap(heading('h2', 'Managed Tiered KV Cache'))
    rehypeHeadingIds()(tree)
    expect(tree.children?.[0].properties?.id).toBe('managed-tiered-kv-cache')
  })

  it('disambiguates duplicate slugs with -1 / -2 suffixes', () => {
    const tree = wrap(
      heading('h2', 'Intro'),
      heading('h2', 'Intro'),
      heading('h2', 'Intro')
    )
    rehypeHeadingIds()(tree)
    const ids = tree.children!.map((c) => c.properties?.id)
    expect(ids).toEqual(['intro', 'intro-1', 'intro-2'])
  })

  it('overwrites pre-existing ids so harden-rewritten user-content-* ids are replaced', () => {
    const node = heading('h2', 'Real Heading')
    node.properties = { id: 'user-content-real-heading' }
    const tree = wrap(node)
    rehypeHeadingIds()(tree)
    expect(tree.children?.[0].properties?.id).toBe('real-heading')
  })

  it('does not stamp non-heading elements', () => {
    const tree = wrap({
      type: 'element',
      tagName: 'p',
      properties: {},
      children: [{ type: 'text', value: 'not a heading' }],
    })
    rehypeHeadingIds()(tree)
    expect(tree.children?.[0].properties?.id).toBeUndefined()
  })
})

describe('findHeadingByHash', () => {
  function renderToContainer(markup: string): HTMLElement {
    const container = document.createElement('div')
    container.innerHTML = markup
    document.body.appendChild(container)
    return container
  }

  it('returns null when the fragment is empty', () => {
    const c = renderToContainer('<h2 id="foo">Foo</h2>')
    expect(findHeadingByHash(c, '#')).toBeNull()
  })

  it('matches an exact id', () => {
    const c = renderToContainer(
      '<h2 id="managed-tiered-kv-cache">Managed Tiered KV Cache</h2>'
    )
    const heading = findHeadingByHash(c, '#managed-tiered-kv-cache')
    expect(heading).not.toBeNull()
    expect(heading?.textContent).toBe('Managed Tiered KV Cache')
  })

  it('falls back to a decoration-insensitive match', () => {
    const c = renderToContainer(
      '<h2 id="section-1-introduction">Section 1 — Introduction</h2>'
    )
    // Author wrote the href without the decoration: #section1introduction
    const heading = findHeadingByHash(c, '#section1introduction')
    expect(heading?.textContent).toBe('Section 1 — Introduction')
  })

  it('falls back to a leading-section-number-stripped match', () => {
    const c = renderToContainer('<h2 id="problem">## Problem</h2>')
    const heading = findHeadingByHash(c, '#3-problem')
    expect(heading?.textContent).toBe('## Problem')
  })

  it('returns null when nothing plausibly matches', () => {
    const c = renderToContainer('<h2 id="real">Real</h2>')
    expect(findHeadingByHash(c, '#ghost-heading')).toBeNull()
  })

  it('does not crash on malformed percent-escapes (#81055)', () => {
    const c = renderToContainer('<h2 id="real">Real</h2>')
    // decodeURIComponent would throw — the helper returns the raw string.
    expect(() => findHeadingByHash(c, '#bad%ZZescape')).not.toThrow()
  })
})