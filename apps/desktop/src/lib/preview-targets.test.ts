import { describe, expect, it } from 'vitest'

import {
  extractPreviewTargets,
  markdownArtifactFileTarget,
  markdownArtifactHref,
  markdownArtifactTargetFromHref,
  previewTargetFromMarkdownHref,
  remarkMarkdownArtifactLinks,
  stripPreviewTargets
} from './preview-targets'

describe('preview target detection', () => {
  it('does not infer preview targets from raw paths or URLs', () => {
    expect(extractPreviewTargets('Preview: http://localhost:5173/')).toEqual([])
    expect(extractPreviewTargets('Open index.html\n/tmp/demo.html\nhttp://localhost:5173/')).toEqual([])
  })

  it('decodes preview markdown hrefs', () => {
    expect(previewTargetFromMarkdownHref('#preview/%2Ftmp%2Fdemo.html')).toBe('/tmp/demo.html')
    expect(previewTargetFromMarkdownHref('#preview:%2Ftmp%2Fdemo.html')).toBe('/tmp/demo.html')
    expect(previewTargetFromMarkdownHref('#media:%2Ftmp%2Fdemo.mp4')).toBeNull()
  })

  it('routes only local Markdown hrefs into the artifact preview', () => {
    expect(markdownArtifactTargetFromHref('/tmp/brief.md')).toBe('/tmp/brief.md')
    expect(markdownArtifactTargetFromHref('./notes/brief.markdown#decision')).toBe('./notes/brief.markdown#decision')
    expect(markdownArtifactTargetFromHref('file:///tmp/brief%20notes.md')).toBe('file:///tmp/brief%20notes.md')
    expect(markdownArtifactTargetFromHref('C:\\Users\\Rohit\\brief.md')).toBe('C:\\Users\\Rohit\\brief.md')

    expect(markdownArtifactTargetFromHref('https://example.com/brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('//example.com/brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('javascript:alert(1).md')).toBeNull()
    expect(markdownArtifactTargetFromHref('data:text/plain,brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('mailto:brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('vbscript:brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('#brief.md')).toBeNull()
    expect(markdownArtifactTargetFromHref('/tmp/brief.txt')).toBeNull()
  })

  it('removes navigation suffixes before resolving artifact files', () => {
    expect(markdownArtifactFileTarget('./notes/brief.markdown#decision')).toBe('./notes/brief.markdown')
    expect(markdownArtifactFileTarget('/tmp/brief.md?view=source')).toBe('/tmp/brief.md')
    expect(markdownArtifactFileTarget('file:///tmp/brief%20notes.md#decision')).toBe('file:///tmp/brief%20notes.md')
  })

  it('rewrites parsed Markdown link and reference destinations without touching code, images, or web URLs', () => {
    const direct = { children: [], type: 'link', url: 'file:///tmp/Q1(report).md' }
    const definition = { identifier: 'brief', type: 'definition', url: 'file:///tmp/reference.markdown' }
    const image = { alt: 'diagram', type: 'image', url: 'file:///tmp/image.md' }
    const inlineCode = { type: 'inlineCode', value: '[brief](file:///tmp/code.md)' }
    const protocolRelative = { children: [], type: 'link', url: '//example.com/file.md' }
    const tree = { children: [direct, definition, image, inlineCode, protocolRelative], type: 'root' }

    remarkMarkdownArtifactLinks()(tree)

    expect(direct.url).toBe(markdownArtifactHref('file:///tmp/Q1(report).md'))
    expect(definition.url).toBe(markdownArtifactHref('file:///tmp/reference.markdown'))
    expect(image.url).toBe('file:///tmp/image.md')
    expect(inlineCode.value).toBe('[brief](file:///tmp/code.md)')
    expect(protocolRelative.url).toBe('//example.com/file.md')
  })

  it('extracts preview targets from already-rendered preview markers', () => {
    expect(extractPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)')).toEqual(['/tmp/demo.html'])
  })

  it('strips preview targets from visible assistant text', () => {
    expect(stripPreviewTargets('ready\n/tmp/mycelium-bunnies.html\nopen it')).toBe(
      'ready\n/tmp/mycelium-bunnies.html\nopen it'
    )
    expect(stripPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)\nopen it')).toBe('open it')
  })
})
