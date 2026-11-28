import { describe, expect, it } from 'vitest'

import { mediaName } from '@/lib/media'
import { $previewStatusBySession, recordPreviewArtifact } from '@/store/preview-status'

import {
  extractPreviewTargets,
  previewArtifactKey,
  previewDisplayLabel,
  previewMarkdownHref,
  previewName,
  previewTargetFromMarkdownHref,
  stripPreviewTargets
} from './preview-targets'

describe('previewName', () => {
  it('shares readable basenames across previews and media without changing target identity', () => {
    const cases = [
      ['https://example.com/my%20report.png?download=1#view', 'my report.png'],
      ['https://example.com/测试.png', '测试.png'],
      ['https://example.com/%E6%B5%8B%E8%AF%95.png', '测试.png'],
      ['https://example.com/a+b.png', 'a+b.png'],
      ['https://example.com/a%2520b.png', 'a%20b.png'],
      ['https://example.com/a%20b%2Fc.png', 'a b%2Fc.png'],
      ['https://example.com/a%20b%5Cc.png', 'a b%5Cc.png'],
      ['https://example.com/a%2fb%5cc.png', 'a%2fb%5cc.png'],
      ['https://example.com/a%252Fb%255Cc.png', 'a%2Fb%5Cc.png'],
      ['file:///tmp/a%20b%2Fc.png', 'a b%2Fc.png'],
      ['file:///tmp/a%20b%5Cc.png', 'a b%5Cc.png'],
      ['file:///tmp/测试.png', '测试.png'],
      ['file:///tmp/a%2520b.png', 'a%20b.png'],
      ['https://example.com/bad%ZZ.png', 'bad%ZZ.png'],
      ['https://example.com/bad%2.png', 'bad%2.png'],
      ['https://example.com/100%.png', '100%.png'],
      ['https://example.com/bad%E6%96.png', 'bad%E6%96.png'],
      ['https://example.com/bad%FF.png', 'bad%FF.png'],
      ['file:///tmp/bad%FF.png', 'bad%FF.png'],
      ['C:\\\\example\\\\测试%20.png', '测试%20.png'],
      ['C:/example/测试%20.png', '测试%20.png'],
      ['\\\\\\\\server\\\\share\\\\测试.png', '测试.png'],
      ['/tmp/测试%20.png', '测试%20.png']
    ]

    for (const [target, name] of cases) {
      expect.soft(mediaName(target), target).toBe(name)
      expect.soft(previewName(target), target).toBe(name)
      expect.soft(previewDisplayLabel(target), target).toBe(`Preview: ${name}`)
      expect.soft(previewTargetFromMarkdownHref(previewMarkdownHref(target)), target).toBe(target)

      if (target.startsWith('file:') && /%2f/i.test(target)) {
        expect.soft(previewArtifactKey(target, '/tmp'), target).toBe(target)
      }
    }

    expect(previewName('https://example.com/')).toBe('example.com')
    expect(previewName('file:///')).toBe('file:///')

    const previous = $previewStatusBySession.get()
    const targets = ['https://one.example/my%20report.png', 'https://two.example/my report.png']

    try {
      for (const target of targets) {
        recordPreviewArtifact('filename-regression', target, '')
      }

      const items = $previewStatusBySession.get()['filename-regression']
      expect(items.map(item => item.target)).toEqual(targets)
      expect(new Set(items.map(item => item.label)).size).toBe(2)
      expect(items.every(item => item.label !== 'my report.png')).toBe(true)
    } finally {
      $previewStatusBySession.set(previous)
    }
  })
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
