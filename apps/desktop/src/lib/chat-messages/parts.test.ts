import { describe, expect, it } from 'vitest'

import { renderMediaTags } from './parts'

describe('renderMediaTags', () => {
  it('renders a simple unquoted path without spaces', () => {
    const result = renderMediaTags('MEDIA:/home/user/report.pdf')
    expect(result).toContain('report.pdf')
    expect(result).not.toContain('MEDIA:')
  })

  it('renders a quoted path with spaces', () => {
    const result = renderMediaTags('MEDIA:"/home/user/My Document.docx"')
    expect(result).toContain('My Document.docx')
    expect(result).not.toContain('MEDIA:')
  })

  it('renders an unquoted path with interior spaces (#96657)', () => {
    const input = 'MEDIA:/home/hermes/Morten - Nobly Kickoff - Opening and cue cards EN.docx'
    const result = renderMediaTags(input)
    // The full path must be captured — not truncated at the first space
    expect(result).toContain('Morten - Nobly Kickoff - Opening and cue cards EN.docx')
    expect(result).not.toContain('MEDIA:')
    // The entire input should be consumed (no leftover literal text)
    expect(result).toBe(renderMediaTags(input))
  })

  it('renders a backtick-quoted path with spaces', () => {
    const result = renderMediaTags('MEDIA:`/home/user/My File.png`')
    expect(result).toContain('My File.png')
    expect(result).not.toContain('MEDIA:')
  })

  it('renders a single-quote path with spaces', () => {
    const result = renderMediaTags("MEDIA:'/home/user/My Audio.mp3'")
    expect(result).toContain('My Audio.mp3')
    expect(result).not.toContain('MEDIA:')
  })

  it('handles a MEDIA tag on its own line', () => {
    const text = `Some text\nMEDIA:/home/user/Report Final.pdf\nMore text`
    const result = renderMediaTags(text)
    expect(result).toContain('Report Final.pdf')
    expect(result).toContain('Some text')
    expect(result).toContain('More text')
    expect(result).not.toContain('MEDIA:')
  })

  it('handles multiple MEDIA tags on separate lines', () => {
    const text = 'MEDIA:/home/user/First File.pdf\nMEDIA:/home/user/Second Image.png'
    const result = renderMediaTags(text)
    expect(result).toContain('First File.pdf')
    expect(result).toContain('Second Image.png')
    expect(result).not.toContain('MEDIA:')
  })

  it('renders a Windows path with spaces', () => {
    const result = renderMediaTags(
      'MEDIA:C:\\Users\\Morten\\My Report.docx'
    )
    expect(result).toContain('My Report.docx')
    expect(result).not.toContain('MEDIA:')
  })

  it('renders a ~/-relative path with spaces', () => {
    const result = renderMediaTags('MEDIA:~/Documents/My Notes.md')
    expect(result).toContain('My Notes.md')
    expect(result).not.toContain('MEDIA:')
  })

  it('leaves non-MEDIA text intact', () => {
    const result = renderMediaTags('Just some regular text without any media tags.')
    expect(result).toBe('Just some regular text without any media tags.')
  })
})