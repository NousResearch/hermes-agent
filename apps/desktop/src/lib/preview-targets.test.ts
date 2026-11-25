import { describe, expect, it } from 'vitest'

import {
  extractPreviewTargets,
  previewArtifactKey,
  previewName,
  previewTargetFromMarkdownHref,
  stripPreviewTargets
} from './preview-targets'

describe('previewName', () => {
  // #85132: `new URL('C:\\...')` parses the drive letter as a scheme, which
  // labelled the tag with the whole backslash path.
  it.each([
    'C:\\Users\\me\\report.html',
    'C:/Users/me/report.html',
    '\\\\server\\share\\report.html',
    '/Users/me/report.html'
  ])('labels %s by its file name', target => {
    expect(previewName(target)).toBe('report.html')
  })

  it.each([
    ['file:///C:/work%20tree/report%20%231.html', 'report #1.html'],
    ['file://server/share/report.html', 'report.html'],
    ['file:///srv/100%.txt', '100%.txt']
  ])('labels file URL %s by its file name', (url, name) => {
    expect(previewName(url)).toBe(name)
  })
})

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

  it('extracts preview targets from already-rendered preview markers', () => {
    expect(extractPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)')).toEqual(['/tmp/demo.html'])
  })

  it('strips preview targets from visible assistant text', () => {
    expect(stripPreviewTargets('ready\n/tmp/mycelium-bunnies.html\nopen it')).toBe(
      'ready\n/tmp/mycelium-bunnies.html\nopen it'
    )
    expect(stripPreviewTargets('[Preview: demo.html](#preview:%2Ftmp%2Fdemo.html)\nopen it')).toBe('\nopen it')
  })
})

describe('previewArtifactKey', () => {
  it.each([
    ['file:///srv/report%20%231.html', '/srv/report #1.html'],
    ['file:///C:/work%20tree/report.html', 'C:\\work tree\\report.html'],
    ['file:///C:/work%20tree/report.html', 'C:/work tree/report.html'],
    ['file://server/share/report.html', '\\\\server\\share\\report.html'],
    ['file:////srv/share/report.html', '//srv/share/report.html'],
    // `\` separates only in drive and UNC paths; a POSIX name may contain one.
    ['file:///srv/name%5Cdraft.html', '/srv/name\\draft.html']
  ])('keys %s as the same artifact as %s', (url, path) => {
    expect(previewArtifactKey(url, '/work')).toBe(previewArtifactKey(path, '/work'))
  })

  it.each([
    'file:///srv/project/..%2f..%2fetc/passwd',
    'file:///srv/project/..%2F..%2Fetc/passwd',
    'file:///C:/work/..%5c..%5cWindows/win.ini',
    'file://server/share/..%5Csecret.txt',
    'file:///tmp/100%.txt'
  ])('leaves %s undecoded', url => {
    expect(previewArtifactKey(url, '/work')).toBe(url)
  })
})
