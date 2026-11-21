import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'

import { preprocessWithTailRepair } from './markdown-streaming'

/**
 * The streaming pipeline's job is to make the href the renderer finally
 * resolves equal the href the sender wrote. A `~` is a legal URL character
 * (userinfo, path segment, query value), so a URL that carries one must reach
 * the parser untouched — not with a stray backslash in front of it, and not
 * percent-escaped.
 *
 * These assertions parse the preprocessed text with the same remark pipeline
 * the renderer uses and compare the RESOLVED href against the authored one.
 */

type MdNode = { children?: MdNode[]; identifier?: string; type: string; url?: string; value?: string }

function parse(markdown: string): MdNode {
  return unified().use(remarkParse).use(remarkGfm).parse(markdown) as MdNode
}

// Every href the parser resolves, reference links included, in document order.
function hrefs(markdown: string): string[] {
  const tree = parse(markdown)
  const definitions = new Map<string, string>()
  const found: (() => string)[] = []

  const walk = (node: MdNode) => {
    if (node.type === 'definition') {
      definitions.set(node.identifier ?? '', node.url ?? '')
    } else if (node.type === 'link' || node.type === 'image') {
      found.push(() => node.url ?? '')
    } else if (node.type === 'linkReference' || node.type === 'imageReference') {
      found.push(() => definitions.get(node.identifier ?? '') ?? '')
    }

    node.children?.forEach(walk)
  }

  walk(tree)

  return found.map(resolve => resolve())
}

// Every node type the parser produced, in document order.
function nodeTypes(markdown: string): string[] {
  const types: string[] = []

  const walk = (node: MdNode) => {
    types.push(node.type)
    node.children?.forEach(walk)
  }

  walk(parse(markdown))

  return types
}

// The plain text the parser sees, with markdown syntax removed.
function text(markdown: string): string {
  const parts: string[] = []

  const walk = (node: MdNode) => {
    if (node.type === 'text' || node.type === 'inlineCode') {
      parts.push(node.value ?? '')
    }

    node.children?.forEach(walk)
  }

  walk(parse(markdown))

  return parts.join('')
}

describe('preprocessWithTailRepair / URLs keep their tildes', () => {
  it.each([
    ['bare URL', 'https://example.com/~alice/report~draft', ['https://example.com/~alice/report~draft']],
    [
      'angle-bracket autolink',
      '<https://example.com/~alice/report~draft>',
      ['https://example.com/~alice/report~draft']
    ],
    [
      'markdown link destination',
      '[Report](https://example.com/~alice/report~draft)',
      ['https://example.com/~alice/report~draft']
    ],
    ['several tildes in one path', 'https://example.com/~a/b~c/d~e', ['https://example.com/~a/b~c/d~e']],
    [
      'tilde in the query string',
      'https://example.com/search?q=~alice~bob&page=2',
      ['https://example.com/search?q=~alice~bob&page=2']
    ],
    ['tilde in the fragment', 'https://example.com/docs#section~2~final', ['https://example.com/docs#section~2~final']],
    ['tilde in userinfo', 'https://~user@example.com/path', ['https://~user@example.com/path']],
    [
      'autolink beside a labeled link',
      '<https://example.com/~a/b> and [x](https://example.com/~c/d)',
      ['https://example.com/~a/b', 'https://example.com/~c/d']
    ]
  ])('resolves the authored href for a %s', (_label, input, expected) => {
    expect(hrefs(preprocessWithTailRepair(input))).toEqual(expected)
  })
})

describe('preprocessWithTailRepair / prose and incomplete-markdown repair survive', () => {
  it.each([
    ['numeric ranges in prose', '20~25 apples and 30~35 pears', '20~25 apples and 30~35 pears'],
    ['an already escaped tilde', '20\\~25 apples', '20~25 apples'],
    ['a tilde before a word', 'approx~value here', 'approx~value here']
  ])('keeps %s as prose', (_label, input, visible) => {
    const output = preprocessWithTailRepair(input)

    expect(text(output)).toBe(visible)
  })

  it.each([
    ['complete strikethrough', '~~deleted~~', 'delete'],
    ['half-streamed strikethrough', '~~deleted', 'delete'],
    ['half-streamed bold', '**bold', 'strong'],
    ['half-streamed italic', '*italic', 'emphasis'],
    ['half-streamed inline code', '`code', 'inlineCode']
  ])('still repairs %s', (_label, input, expectedType) => {
    expect(nodeTypes(preprocessWithTailRepair(input))).toContain(expectedType)
  })

  it('repairs the tail without changing earlier paragraphs', () => {
    const input = ['First paragraph with a URL https://example.com/~a/b.', '', 'Second paragraph **bold'].join('\n')

    const output = preprocessWithTailRepair(input)

    expect(hrefs(output)).toEqual(['https://example.com/~a/b'])
    expect(nodeTypes(output)).toContain('strong')
  })
})
