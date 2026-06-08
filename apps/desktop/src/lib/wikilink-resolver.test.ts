import { describe, expect, it } from 'vitest'

import {
  basename,
  buildVaultIndex,
  dirname,
  resolveObsidianLinks,
  resolveRelativeMdLinks,
  resolveWikilinks
} from './wikilink-resolver'

describe('wikilink-resolver', () => {
  describe('dirname', () => {
    it('returns parent directory for Unix paths', () => {
      expect(dirname('/Users/test/vault/note.md')).toBe('/Users/test/vault')
    })

    it('returns parent directory for nested paths', () => {
      expect(dirname('/a/b/c/d.md')).toBe('/a/b/c')
    })

    it('handles root path', () => {
      expect(dirname('/note.md')).toBe('/note.md')
    })
  })

  describe('basename', () => {
    it('extracts filename without extension', () => {
      expect(basename('/Users/test/note.md', '.md')).toBe('note')
    })

    it('extracts filename with spaces', () => {
      expect(basename('/vault/My Note.md', '.md')).toBe('My Note')
    })

    it('extracts full filename without ext arg', () => {
      expect(basename('/vault/note.md')).toBe('note.md')
    })
  })

  describe('buildVaultIndex', () => {
    it('maps basenames to full paths (lowercase)', () => {
      const index = buildVaultIndex(['/vault/Notes/My Note.md', '/vault/ideas.md'])
      expect(index.byBasename.get('my note')).toBe('/vault/Notes/My Note.md')
      expect(index.byBasename.get('ideas')).toBe('/vault/ideas.md')
    })

    it('first match wins for duplicate basenames', () => {
      const index = buildVaultIndex(['/vault/a/note.md', '/vault/b/note.md'])
      expect(index.byBasename.get('note')).toBe('/vault/a/note.md')
    })
  })

  describe('resolveWikilinks', () => {
    const index = buildVaultIndex(['/vault/My Note.md', '/vault/ideas.md'])

    it('resolves [[Note]] to preview link', () => {
      const result = resolveWikilinks('See [[My Note]] for details.', index)
      expect(result).toContain('[My Note]')
      expect(result).toContain('#preview/')
      expect(result).toContain('My%20Note')
    })

    it('resolves [[Note|alias]] with alias text', () => {
      const result = resolveWikilinks('See [[ideas|my ideas]] here.', index)
      expect(result).toContain('[my ideas]')
      expect(result).toContain('#preview/')
      expect(result).toContain('ideas')
    })

    it('resolves [[Note#Heading]] with heading', () => {
      const result = resolveWikilinks('Jump to [[My Note#Introduction]].', index)
      expect(result).toContain('[My Note > Introduction]')
      expect(result).toContain('#preview/')
    })

    it('unresolved wikilinks become inline code', () => {
      const result = resolveWikilinks('See [[Unknown Note]].', index)
      expect(result).toBe('See `Unknown Note`.')
    })

    it('returns inline code when index is null', () => {
      const result = resolveWikilinks('See [[My Note]].', null)
      expect(result).toBe('See `My Note`.')
    })
  })

  describe('resolveRelativeMdLinks', () => {
    it('resolves relative .md links', () => {
      const result = resolveRelativeMdLinks(
        '[other](other.md)',
        '/vault/notes'
      )
      expect(result).toContain('[other]')
      expect(result).toContain('#preview/')
      expect(result).toContain('other.md')
    })

    it('resolves ../ relative paths', () => {
      const result = resolveRelativeMdLinks(
        '[readme](../README.md)',
        '/vault/notes'
      )
      expect(result).toContain('#preview/')
      expect(result).toContain('README.md')
    })

    it('resolves relative links with heading anchor', () => {
      const result = resolveRelativeMdLinks(
        '[section](other.md#intro)',
        '/vault/notes'
      )
      expect(result).toContain('#intro')
    })

    it('skips absolute URLs', () => {
      const input = '[link](https://example.com)'
      expect(resolveRelativeMdLinks(input, '/vault')).toBe(input)
    })

    it('skips file:// URLs', () => {
      const input = '[link](file:///something.md)'
      expect(resolveRelativeMdLinks(input, '/vault')).toBe(input)
    })

    it('skips anchor-only links', () => {
      const input = '[link](#heading)'
      expect(resolveRelativeMdLinks(input, '/vault')).toBe(input)
    })

    it('skips non-.md links', () => {
      const input = '[link](image.png)'
      expect(resolveRelativeMdLinks(input, '/vault')).toBe(input)
    })

    it('skips image links (prefixed with !)', () => {
      const input = '![image](screenshot.png)'
      expect(resolveRelativeMdLinks(input, '/vault')).toBe(input)
    })
  })

  describe('resolveObsidianLinks', () => {
    const index = buildVaultIndex(['/vault/My Note.md', '/vault/ideas.md'])

    it('resolves both wikilinks and relative .md links', () => {
      const result = resolveObsidianLinks(
        'See [[My Note]] and [other](other.md).',
        '/vault/index.md',
        index
      )
      expect(result).toContain('[My Note]')
      expect(result).toContain('#preview/')
      expect(result).toContain('[other]')
    })

    it('handles mixed resolved and unresolved', () => {
      const result = resolveObsidianLinks(
        '[[My Note]] and [[Unknown]] and [relative](../other.md).',
        '/vault/index.md',
        index
      )
      // My Note is resolved
      expect(result).toContain('[My Note]')
      // Unknown is unresolved → inline code
      expect(result).toContain('`Unknown`')
      // relative link is resolved
      expect(result).toContain('[relative]')
    })
  })
})
