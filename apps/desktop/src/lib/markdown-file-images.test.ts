import { describe, expect, it, vi } from 'vitest'

import {
  createMarkdownFileImageLoader,
  FILE_PREVIEW_IMAGE_PATH_ATTR,
  FILE_PREVIEW_IMAGE_SOURCE_ATTR,
  filePreviewImageRehypePlugin,
  MARKDOWN_FILE_IMAGE_MAX_BYTES,
  resolveMarkdownFileImagePath
} from './markdown-file-images'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

describe('resolveMarkdownFileImagePath', () => {
  it('resolves descendant image paths beside a POSIX markdown file', () => {
    expect(resolveMarkdownFileImagePath('diagrams/chart.svg', '/workspace/docs/report.md')).toBe(
      '/workspace/docs/diagrams/chart.svg'
    )
    expect(resolveMarkdownFileImagePath('./figures/a%23final.svg?cache=1#figure', '/workspace/docs/report.md')).toBe(
      '/workspace/docs/figures/a#final.svg'
    )
  })

  it('preserves Windows drive and UNC markdown path semantics', () => {
    expect(resolveMarkdownFileImagePath('diagrams/chart.svg', 'C:\\workspace\\docs\\report.md')).toBe(
      'C:\\workspace\\docs\\diagrams\\chart.svg'
    )
    expect(resolveMarkdownFileImagePath('diagrams/chart.svg', '\\\\server\\share\\docs\\report.md')).toBe(
      '\\\\server\\share\\docs\\diagrams\\chart.svg'
    )
  })

  it('treats backslash as a filename character in POSIX markdown paths', () => {
    expect(resolveMarkdownFileImagePath('diagrams/chart.svg', '/repo/report\\2026.md')).toBe('/repo/diagrams/chart.svg')
  })

  it.each([
    '../outside.svg',
    'nested/../../outside.svg',
    '%2e%2e/outside.svg',
    '/absolute/chart.svg',
    'file:///tmp/chart.svg',
    '~/chart.svg',
    'C:\\tmp\\chart.svg',
    '\\\\server\\share\\chart.svg',
    'https://example.com/chart.svg',
    'data:image/svg+xml;base64,PHN2Zy8+',
    'diagrams/report.pdf'
  ])('rejects sources outside the descendant-image contract: %s', source => {
    expect(resolveMarkdownFileImagePath(source, '/workspace/docs/report.md')).toBeNull()
  })
})

describe('filePreviewImageRehypePlugin', () => {
  it('marks only a bounded number of descendant images with trusted post-sanitize metadata', () => {
    const tree = {
      type: 'root',
      children: [
        { type: 'element', tagName: 'img', properties: { alt: 'one', src: 'images/one.svg' } },
        { type: 'element', tagName: 'img', properties: { alt: 'web', src: 'https://example.com/two.svg' } },
        { type: 'element', tagName: 'img', properties: { alt: 'three', src: 'images/three.png' } },
        { type: 'element', tagName: 'img', properties: { alt: 'four', src: 'images/four.webp' } }
      ]
    }

    const transform = filePreviewImageRehypePlugin({ markdownPath: '/workspace/docs/report.md', maxImages: 2 })

    transform(tree)

    const properties = tree.children.map(node => node.properties as Record<string, unknown>)

    expect(properties[0][FILE_PREVIEW_IMAGE_PATH_ATTR]).toBe('/workspace/docs/images/one.svg')
    expect(properties[0][FILE_PREVIEW_IMAGE_SOURCE_ATTR]).toBe('/workspace/docs/report.md')
    expect(properties[1][FILE_PREVIEW_IMAGE_PATH_ATTR]).toBeUndefined()
    expect(properties[2][FILE_PREVIEW_IMAGE_PATH_ATTR]).toBe('/workspace/docs/images/three.png')
    expect(properties[3][FILE_PREVIEW_IMAGE_PATH_ATTR]).toBeUndefined()
    expect(properties[3].src).toBe('')
    expect(properties[0].src).toMatch(/^data:image\/gif;base64,/)
  })
})

describe('createMarkdownFileImageLoader', () => {
  it('bounds concurrent reads and passes the dedicated per-image byte ceiling', async () => {
    const reads = [deferred<string>(), deferred<string>(), deferred<string>()]
    const read = vi.fn((_path: string, _source: string, _maxBytes: number) => reads[read.mock.calls.length - 1].promise)
    const loader = createMarkdownFileImageLoader(read)

    const loads = [
      loader.load('/docs/one.png', '/docs/report.md'),
      loader.load('/docs/two.png', '/docs/report.md'),
      loader.load('/docs/three.png', '/docs/report.md')
    ]

    expect(read).toHaveBeenCalledTimes(2)
    expect(read.mock.calls.map(call => call[2])).toEqual([MARKDOWN_FILE_IMAGE_MAX_BYTES, MARKDOWN_FILE_IMAGE_MAX_BYTES])
    reads[0].resolve('data:image/png;base64,one')
    await loads[0]
    await vi.waitFor(() => expect(read).toHaveBeenCalledTimes(3))
    reads[1].resolve('data:image/png;base64,two')
    reads[2].resolve('data:image/png;base64,three')
    await Promise.all(loads.slice(1))
  })

  it('rejects data URLs beyond the document retained-byte budget', async () => {
    const loader = createMarkdownFileImageLoader(async path => `data:image/png;base64,${path}`, {
      maxConcurrent: 1,
      maxRetainedDataUrlBytes: 48
    })

    await expect(loader.load('1234567890', '/docs/report.md')).resolves.toContain('1234567890')
    await expect(loader.load('abcdefghij', '/docs/report.md')).rejects.toThrow('document image budget')
  })

  it('cancels queued and in-flight work on disposal without retaining late results', async () => {
    const active = deferred<string>()
    const read = vi.fn(() => active.promise)
    const loader = createMarkdownFileImageLoader(read, { maxConcurrent: 1 })
    const inFlight = loader.load('/docs/one.png', '/docs/report.md')
    const queued = loader.load('/docs/two.png', '/docs/report.md')

    loader.dispose()
    active.resolve('data:image/png;base64,late')

    await expect(inFlight).rejects.toThrow('disposed')
    await expect(queued).rejects.toThrow('disposed')
    await expect(loader.load('/docs/three.png', '/docs/report.md')).rejects.toThrow('disposed')
    expect(loader.retainedDataUrlBytes()).toBe(0)
    expect(read).toHaveBeenCalledOnce()
  })
})
