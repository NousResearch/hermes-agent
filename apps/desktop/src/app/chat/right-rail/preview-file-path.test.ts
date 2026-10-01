import { describe, expect, it } from 'vitest'

import { filePathForTarget } from './preview-file'

// A preview target reaches the fs layer as a plain path. When it arrives as a
// `file:` URL instead, `URL.pathname` keeps the leading slash the URL form
// always carries, so a Windows path becomes `/C:/Users/...` — which no fs call
// can open. The preview then reports "Text preview failed: file does not
// exist" for a file sitting right there on disk.
describe('filePathForTarget', () => {
  it('strips the leading slash a file URL adds before a Windows drive letter', () => {
    const path = filePathForTarget({
      url: 'file:///C:/Users/Someone/AppData/Local/hermes/scratch/notes.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('C:/Users/Someone/AppData/Local/hermes/scratch/notes.md')
    // The bug shape, named so a regression cannot pass by accident.
    expect(path.startsWith('/')).toBe(false)
  })

  it('keeps the leading slash on a POSIX file URL, which needs it', () => {
    const path = filePathForTarget({
      url: 'file:///home/someone/notes.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('/home/someone/notes.md')
  })

  it('decodes percent-escapes in a Windows path', () => {
    const path = filePathForTarget({
      url: 'file:///C:/Users/Someone/My%20Files/ban-giao.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('C:/Users/Someone/My Files/ban-giao.md')
  })

  it('handles a lowercase drive letter', () => {
    const path = filePathForTarget({
      url: 'file:///d:/data/report.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('d:/data/report.md')
  })

  it('prefers an explicit path over the url', () => {
    const path = filePathForTarget({
      path: 'C:/already/a/path.md',
      url: 'file:///C:/ignored.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('C:/already/a/path.md')
  })

  it('passes a non-file url through untouched', () => {
    const path = filePathForTarget({
      url: 'https://example.com/page'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('https://example.com/page')
  })

  it('falls back to the raw value when the url does not parse', () => {
    const path = filePathForTarget({
      url: 'C:\\Users\\Someone\\notes.md'
    } as Parameters<typeof filePathForTarget>[0])

    expect(path).toBe('C:\\Users\\Someone\\notes.md')
  })
})
