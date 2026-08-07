import { describe, expect, it } from 'vitest'

import { desktopMarkdownHref, desktopTargetFromMarkdownHref, remarkDesktopLinks } from './desktop-links'

describe('desktop response links', () => {
  it.each(['file:///C:/Users/example/My%20Note.md', 'obsidian://open?vault=Personal&file=00%20Inbox%2FMy%20Note.md'])(
    'round-trips an allowed target through a safe fragment: %s',
    target => {
      const href = desktopMarkdownHref(target)

      expect(href).toMatch(/^#desktop-open\//)
      expect(desktopTargetFromMarkdownHref(href ?? undefined)).toBe(target)
    }
  )

  it.each([
    'javascript:alert(1)',
    'vscode://file/C:/Users/example/note.md',
    'obsidian://new?vault=Personal&name=Injected',
    'file://attacker/share/note.md',
    'file:////attacker/share/note.md',
    'file://///attacker/share/note.md',
    'file://localhost//attacker/share/note.md',
    'obsidian://open@attacker?vault=Personal',
    'obsidian://open/other-action?vault=Personal',
    'file:///C:/Users/example/payload.exe',
    'file:///C:/Users/example/payload.cmd',
    'file:///C:/Users/example/payload.bat',
    'file:///C:/Users/example/installer.msi',
    'file:///C:/Users/example/shortcut.lnk',
    'file:///C:/Users/example/website.url',
    'file:///C:/Users/example/script.ps1',
    'file:///C:/Users/example/document.pdf.exe',
    'file:///C:/Users/example/encoded%2Eexe',
    'file:///C:/Users/example/note%2Fescape.md',
    'file:///C:/Users/example/note%5Cescape.md'
  ])('does not wrap a disallowed target: %s', target => {
    expect(desktopMarkdownHref(target)).toBeNull()
  })

  it('rewrites link AST nodes without changing image AST nodes', () => {
    const link = { type: 'link', url: 'file:///C:/tmp/a_(b).md' }
    const image = { type: 'image', url: 'file:///C:/tmp/diagram.png' }
    const tree = { children: [link, image], type: 'root' }

    remarkDesktopLinks()(tree)

    expect(link.url).toMatch(/^#desktop-open\//)
    expect(image.url).toBe('file:///C:/tmp/diagram.png')
  })

  it('rejects malformed and forged safe fragments', () => {
    expect(desktopTargetFromMarkdownHref('#desktop-open/%E0%A4%A')).toBeNull()
    expect(desktopTargetFromMarkdownHref(`#desktop-open/${encodeURIComponent('javascript:alert(1)')}`)).toBeNull()
  })
})
