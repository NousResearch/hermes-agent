import { describe, expect, it } from 'vitest'

import { TOOL_IMAGE_PAGE_SIZE, toolImageSources } from './tool-images'

describe('toolImageSources', () => {
  it('exposes the shared five-thumbnail page size', () => {
    expect(TOOL_IMAGE_PAGE_SIZE).toBe(5)
  })

  it('preserves native crop pixels, order and deduplication ahead of all reference fallbacks', () => {
    const crop = 'data:image/png;base64,Y3JvcA=='

    const output = {
      content: [
        { type: 'text', text: 'MEDIA:/incidental.png' },
        { type: 'image_url', image_url: { url: crop } },
        { type: 'image', data: 'c2Vjb25k', mimeType: 'image/webp' },
        { type: 'image_url', image_url: { url: crop } }
      ],
      image_paths: ['/output.png'],
      screenshot_path: '/screenshot.png'
    }

    expect(toolImageSources({ image_url: '/original.png' }, output)).toEqual([crop, 'data:image/webp;base64,c2Vjb25k'])
  })

  it('collects explicit arrays and flat descriptors without dropping later images', () => {
    const paths = Array.from({ length: 137 }, (_, i) => `/tmp/render ${i}.png`)

    const output = {
      meta: { screenshot_path: '/first.png' },
      screenshot_path: '/second.png',
      images: [
        '/first.png',
        { path: '/third.webp' },
        { url: 'https://example.com/generated?id=2' },
        { image_url: { url: 'https://example.com/generated?id=3' } },
        { type: 'image', mimeType: 'image/png', data: 'YQ==' },
        { image_path: '/fourth.jpg' },
        { image: 'data:image/png;base64,Yg==' }
      ],
      image_paths: paths,
      output: 'Screenshot path: /second.png\nMEDIA:/last.png'
    }

    expect(toolImageSources({ path: '/input.png' }, output)).toEqual([
      '/first.png',
      '/second.png',
      '/third.webp',
      'https://example.com/generated?id=2',
      'https://example.com/generated?id=3',
      'data:image/png;base64,YQ==',
      '/fourth.jpg',
      'data:image/png;base64,Yg==',
      ...paths,
      '/last.png'
    ])
    expect(toolImageSources({ images: paths, image_paths: ['/extra.png', paths[0]] }, {})).toEqual([
      ...paths,
      '/extra.png'
    ])
  })

  it('accepts matching standalone MEDIA quotes while rejecting incidental or malformed markers', () => {
    expect(
      toolImageSources(
        {},
        {
          result: [
            'MEDIA:"/tmp/double quoted.png"',
            "MEDIA:'/tmp/single quoted.jpg'",
            '  MEDIA: `/tmp/backtick.webp`  ',
            'MEDIA:/tmp/plain.png',
            'prose MEDIA:/tmp/incidental.png',
            'MEDIA:"/tmp/mismatched.png\'',
            'MEDIA:`/tmp/unclosed.png',
            'MEDIA:"/tmp/file.png" trailing prose',
            'MEDIA:/tmp/not-an-image.pdf'
          ].join('\n')
        }
      )
    ).toEqual(['/tmp/double quoted.png', '/tmp/single quoted.jpg', '/tmp/backtick.webp', '/tmp/plain.png'])
  })

  it('unwraps one structured MCP result and prefers its native pixels over outer references', () => {
    expect(toolImageSources({}, { result: { images: ['/mcp.png'], image_paths: ['/other.png'] } })).toEqual([
      '/mcp.png',
      '/other.png'
    ])
    expect(
      toolImageSources(
        { path: '/input.png' },
        {
          screenshot_path: '/outer.png',
          result: { content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,YQ==' } }] }
        }
      )
    ).toEqual(['data:image/png;base64,YQ=='])
  })

  it('normalizes JSON object strings once and invalidates memoization on immutable live updates', () => {
    const input = { image_paths: ['/pending.png'] }
    const output = { result: { images: ['/finished.png'] } }
    const expected = ['/finished.png']

    expect(toolImageSources(input, undefined)).toEqual(['/pending.png'])
    expect(toolImageSources(input, output)).toEqual(expected)
    expect(toolImageSources(JSON.stringify(input), JSON.stringify(output))).toEqual(expected)
    expect(toolImageSources(input, JSON.stringify(output))).toEqual(expected)
    expect(toolImageSources(JSON.stringify(input), undefined)).toEqual(['/pending.png'])
    const sources = toolImageSources(input, output)
    expect(toolImageSources(input, output)).toBe(sources)
    expect(toolImageSources(input, { result: { images: ['/new.png'] } })).toEqual(['/new.png'])
    expect(toolImageSources({ image_paths: ['/new-input.png'] }, {})).toEqual(['/new-input.png'])

    for (const invalid of ['{broken', 'null', '42', '["/array.png"]', 'MEDIA:/raw.png']) {
      expect(toolImageSources(undefined, invalid)).toEqual([])
    }
  })

  it('never recursively mines external objects, arbitrary keys, text blocks or nested arrays', () => {
    const cyclic: Record<string, unknown> = {}
    cyclic.result = cyclic
    expect(toolImageSources({}, cyclic)).toEqual([])
    expect(
      toolImageSources(
        {},
        {
          result: { result: { images: ['/too-deep.png'] } },
          data: { images: ['/external.png'] },
          page: { image_url: 'https://example.com/page-asset.png' },
          content: [{ type: 'text', text: 'MEDIA:/page-text.png' }],
          text: 'https://example.com/incidental.png',
          images: [['/nested.png'], { nested: { path: '/nested-object.png' } }],
          image_paths: [{ path: '/not-a-path-string.png' }]
        }
      )
    ).toEqual([])
  })

  it('rejects unsafe schemes and non-image references in every explicit format', () => {
    const invalid = [
      'javascript:alert(1).png',
      'vbscript:alert(1).png',
      'blob:https://example.com/x.png',
      'ftp://example.com/x.png',
      'data:text/html;base64,YQ==',
      '/tmp/settings.yaml',
      '/tmp/movie.mp4',
      'java\nscript:alert(1).png'
    ]

    for (const source of invalid) {
      expect(
        toolImageSources(
          { image_url: source },
          {
            images: [source, { path: source }, { image_url: { url: source } }],
            image_paths: [source],
            result: { image_path: source }
          }
        )
      ).toEqual([])
    }

    expect(
      toolImageSources(
        {},
        {
          images: [null, 42, { type: 'text', url: '/not-an-image-block.png' }],
          content: [{ type: 'image', mimeType: 'text/html', data: 'YQ==' }]
        }
      )
    ).toEqual([])
    expect(toolImageSources({}, { images: ['C:\\renders\\one.png', 'file:///tmp/two.jpg', './three.gif'] })).toEqual([
      'C:\\renders\\one.png',
      'file:///tmp/two.jpg',
      './three.gif'
    ])
  })
})
