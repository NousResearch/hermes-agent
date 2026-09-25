import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

/**
 * remark-gfm accepts a single `~` as a strikethrough delimiter, so prose
 * ranges like `1~10,11~20` paired two tildes and rendered the middle as
 * deleted. Lone tildes in prose must survive preprocessing; intentional
 * `~~…~~` strikethrough, inline code, and code fences must be untouched.
 */

type MdNode = { children?: MdNode[]; type: string; value?: string }

function parse(markdown: string): MdNode {
  return unified().use(remarkParse).use(remarkGfm).parse(preprocessMarkdown(markdown)) as MdNode
}

function deleteNodes(node: MdNode): MdNode[] {
  const found: MdNode[] = []

  const walk = (current: MdNode) => {
    if (current.type === 'delete') {
      found.push(current)
    }

    current.children?.forEach(walk)
  }

  walk(node)

  return found
}

function allText(node: MdNode): string {
  return [node.value ?? '', ...(node.children ?? []).map(allText)].join('')
}

describe('preprocessMarkdown / lone tilde escaping', () => {
  it('keeps numeric ranges like 1~10,11~20 as literal text', () => {
    const input = 'Rows 1~10,11~20 are affected.'

    expect(deleteNodes(parse(input))).toEqual([])
    expect(allText(parse(input))).toContain('1~10,11~20')
  })

  it('still renders intentional double-tilde strikethrough', () => {
    const input = '~~struck~~ stays'

    expect(deleteNodes(parse(input))).toHaveLength(1)
    expect(allText(parse(input))).toContain('struck')
  })

  it('never escapes tildes inside inline code', () => {
    const input = 'Path `~/x~y` is code.'

    expect(preprocessMarkdown(input)).toBe(input)
    expect(deleteNodes(parse(input))).toEqual([])
  })

  it('never escapes tildes inside code fences', () => {
    const input = '```\na~b\nc~d\n```'

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('leaves already-escaped tildes alone', () => {
    const input = 'an \\~ escaped tilde'

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('keeps lone tilde pairing from crossing inline-code boundaries', () => {
    const input = 'a ~ `code` ~ b'

    expect(deleteNodes(parse(input))).toEqual([])
  })
})
