import { describe, expect, it } from 'vitest'

import { folderMarkdownHref, folderPathFromMarkdownHref } from './folder-links'

describe('explicit folder hrefs', () => {
  it('round-trips absolute paths once with Markdown-safe encoding', () => {
    for (const path of ['C:/Reports', 'C:\\Reports\\Проект #1 50% (final)', '/tmp/日本語/a(b)/%2F']) {
      const href = folderMarkdownHref(path)
      expect(href.startsWith('#folder/')).toBe(true)
      expect(href).not.toMatch(/[()\s]/)
      expect(folderPathFromMarkdownHref(href)).toBe(path)
    }
  })

  it('rejects malformed encoding and non-absolute or network payloads without reinterpretation', () => {
    for (const href of [
      undefined,
      '#section',
      '#folder/',
      '#folder/%',
      '#folder/%C0%AF',
      '#folder/C:/Reports',
      ...[
        'relative',
        '~/Reports',
        'C:Reports',
        'file:///tmp',
        '//server/share',
        '\\\\server\\share',
        '\\\\?\\C:\\Reports',
        '/tmp/\0x'
      ].map(p => `#folder/${encodeURIComponent(p)}`)
    ]) {
      expect(folderPathFromMarkdownHref(href)).toBeNull()
    }
  })
})
