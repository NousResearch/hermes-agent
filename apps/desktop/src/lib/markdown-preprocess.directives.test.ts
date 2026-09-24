import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

/**
 * A transcript directive (`::onboarding{...}`) is a card only while the parser
 * hands the paragraph over as ONE text node; the moment an attribute value
 * reads as markdown the paragraph splits into element children and the raw
 * directive paints as the user's message. These are the shapes that leaked.
 */

// The paragraph's children as the renderer will see them, after preprocess.
function paragraphChildren(markdown: string) {
  const tree = unified().use(remarkParse).use(remarkGfm).parse(preprocessMarkdown(markdown)) as {
    children: { children?: { type: string; value?: string }[] }[]
  }

  return tree.children[0]?.children ?? []
}

const briefs = [
  'Track monarch sightings *by week* and plot them',
  'Track a_b and c_d values across runs',
  'Sync ~/notes to ~/backup nightly',
  'Wrap the `run` command with retries',
  'Compare <before> and <after> snapshots',
  'Parse [id] tokens and \\n escapes'
]

describe('preprocessMarkdown / directive lines', () => {
  it.each(briefs)('stays one text node with brief: %s', brief => {
    const line = `::onboarding{step="handoff" task="Tracker" brief="${brief}" plan="build"}`
    const children = paragraphChildren(line)

    expect(children.map(child => child.type)).toEqual(['text'])
    // The parser eats the escapes, so the directive arrives as written.
    expect(children[0]?.value).toBe(line)
  })

  it('leaves the surrounding prose to markdown', () => {
    const text = 'Perfect, that is *all* I needed.\n\n::onboarding{step="handoff" task="T" brief="a_b" plan="build"}'

    const tree = unified().use(remarkParse).use(remarkGfm).parse(preprocessMarkdown(text)) as {
      children: { children?: { type: string }[] }[]
    }

    expect(tree.children[0]?.children?.map(child => child.type)).toEqual(['text', 'emphasis', 'text'])
    expect(tree.children[1]?.children?.map(child => child.type)).toEqual(['text'])
  })

  it('does not touch a directive inside a code fence', () => {
    const text = '```\n::onboarding{step="handoff" brief="a_b"}\n```'

    expect(preprocessMarkdown(text)).toBe(text)
  })

  it('does not touch prose that merely contains ::', () => {
    const text = 'Use std::vector<int> for *speed*.'

    expect(preprocessMarkdown(text)).toBe(text)
  })
})
