import { describe, expect, it } from 'vitest'

import { obsidianHrefFromMarkdownHref, remarkObsidianLinks } from './obsidian-link'

const OBSIDIAN_URL =
  'obsidian://open?vault=PG%20Vault&file=Social%20Media%2FDrafts%2F2026-08-21-americana-vin-fiz-parts-train.md'

describe('Obsidian Markdown link transport', () => {
  it('encodes an Obsidian link as an inert fragment and restores it exactly', () => {
    const link = { type: 'link', url: OBSIDIAN_URL, children: [{ type: 'text' }] }
    const tree = { type: 'root', children: [link] }

    remarkObsidianLinks()(tree)

    expect(link.url).toMatch(/^#obsidian:/)
    expect(obsidianHrefFromMarkdownHref(link.url)).toBe(OBSIDIAN_URL)
  })

  it('ignores normal links and rejects malformed or spoofed fragments', () => {
    const link = { type: 'link', url: 'https://example.com', children: [] }

    remarkObsidianLinks()({ type: 'root', children: [link] })

    expect(link.url).toBe('https://example.com')
    expect(obsidianHrefFromMarkdownHref('#obsidian:not-a-custom-uri')).toBeNull()
    expect(obsidianHrefFromMarkdownHref('#obsidian:%E0%A4%A')).toBeNull()
  })
})
