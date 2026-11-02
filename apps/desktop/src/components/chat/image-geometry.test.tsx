import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { MarkdownImage, MessageTextContent } from '@/components/assistant-ui/markdown-text'
import { $connection } from '@/store/session'

import { GeneratedImage } from './generated-image-result'

vi.mock('./image-generation-placeholder', () => ({ DiffusionCanvas: () => null }))

const paths = { first: '/geometry/first.svg', second: '/geometry/second.svg' }
const data = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800"/>'
let reads: Array<{ resolve: (value: string) => void; reject: (error: Error) => void }>
const read = vi.fn(() => new Promise<string>((resolve, reject) => reads.push({ resolve, reject })))

beforeEach(() => {
  reads = []
  read.mockClear()
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: read, api: vi.fn(() => read().then(dataUrl => ({ dataUrl }))) })
  $connection.set({ connectionId: 'geometry-local', mode: 'local', profile: 'a' } as never)
})
afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.unstubAllGlobals()
})

const cases = [
  { kind: 'markdown', hint: undefined },
  { kind: 'media', hint: undefined },
  ...[undefined, 'landscape', 'square'].map(hint => ({ kind: 'generated', hint })),
  { kind: 'markdown', hint: 'intrinsic' },
  { kind: 'generated', hint: 'intrinsic' }
]

function content(kind: string, path: string, hint?: string) {
  if (kind === 'generated') {
    return (
      <GeneratedImage
        aspectRatio={hint}
        result={{ success: true, image: path, pixel_size: hint === 'intrinsic' ? '800x800' : undefined }}
      />
    )
  }

  if (kind === 'media') {
    return <MessageTextContent text={`MEDIA:${path}`} />
  }

  return (
    <MarkdownImage
      alt="geometry"
      height={hint === 'intrinsic' ? 800 : undefined}
      src={path}
      width={hint === 'intrinsic' ? 800 : undefined}
    />
  )
}

function frame(container: HTMLElement) {
  return container.querySelector<HTMLElement>('[data-slot="aui_generated-image"], [data-slot="aui_markdown-image"]')!
}

async function decode(container: HTMLElement, width = 800, height = 800) {
  await act(async () => reads.shift()!.resolve(data))
  const img = container.querySelector('img')!
  Object.defineProperties(img, { naturalWidth: { value: width }, naturalHeight: { value: height } })
  fireEvent.load(img)
}

it.each(cases)('reserves $kind frames through cold decode, warm return and failure ($hint)', async ({ kind, hint }) => {
  // Unique paths keep parameter rows independent without a production cache-reset API.
  const path = `${paths.first}?${kind}-${hint}`
  const first = render(content(kind, path, hint))
  const cold = frame(first.container)
  expect(cold).not.toBeNull()
  const coldSize = cold.style.cssText

  if (hint === 'intrinsic') {
    expect(parseFloat(cold.style.aspectRatio)).toBe(1)
  }

  await decode(first.container)
  expect(frame(first.container).style.cssText).toBe(coldSize)
  first.unmount()

  const warm = render(content(kind, path, hint))
  const reserved = frame(warm.container)
  expect(parseFloat(reserved.style.aspectRatio)).toBe(1)
  const warmSize = reserved.style.cssText
  await decode(warm.container)
  expect(frame(warm.container).style.cssText).toBe(warmSize)

  // A broken image collapses to its text line instead of an empty frame, and
  // the next markdown mount of that source does not reserve a frame just to
  // collapse it. A generated image's children are all absolutely positioned,
  // so it keeps its hinted frame: a successful retry must still be visible.
  fireEvent.error(warm.container.querySelector('img')!)
  expect(warm.container.textContent).toMatch(/Open image/i)
  expect(frame(warm.container)?.style.aspectRatio ?? '').toBe('')
  warm.unmount()

  const retry = render(content(kind, path, hint))

  if (kind !== 'generated' && hint !== 'intrinsic') {
    expect(frame(retry.container)?.style.aspectRatio ?? '').toBe('')
  }

  await decode(retry.container)
  expect(retry.container.querySelector('img')).not.toBeNull()

  if (kind === 'generated') {
    expect(parseFloat(frame(retry.container).style.aspectRatio)).toBeGreaterThan(0)
  }
})

it('keeps the pending generated-image frame when its result arrives', async () => {
  const path = '/geometry/pending.svg'
  const mounted = render(<GeneratedImage aspectRatio="square" />)
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBe(1)
  // The hint can arrive after the pending card mounted; nothing is shown yet.
  mounted.rerender(<GeneratedImage aspectRatio="landscape" />)
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBeCloseTo(16 / 9)
  const pendingStyle = frame(mounted.container).style.cssText
  mounted.rerender(
    <GeneratedImage aspectRatio="landscape" result={{ success: true, image: path, pixel_size: '900x600' }} />
  )

  // Must hold before the file read settles, not just after img.onload.
  expect(frame(mounted.container).style.cssText).toBe(pendingStyle)
  await decode(mounted.container, 900, 600)
  expect(frame(mounted.container).style.cssText).toBe(pendingStyle)
})

it('keys inline data sources by a bounded digest so they are remembered', async () => {
  const big = (fill: string) => `data:image/png;base64,${fill.repeat(200_000)}`
  const first = render(<MarkdownImage alt="inline" src={big('A')} />)
  expect(first.container.querySelector('img')).not.toBeNull()
  const img = first.container.querySelector('img')!
  Object.defineProperties(img, { naturalWidth: { value: 900 }, naturalHeight: { value: 600 } })
  fireEvent.load(img)
  first.unmount()

  expect(parseFloat(frame(render(<MarkdownImage alt="inline" src={big('A')} />).container).style.aspectRatio)).toBe(1.5)
  expect(parseFloat(frame(render(<MarkdownImage alt="inline" src={big('B')} />).container).style.aspectRatio)).not.toBe(
    1.5
  )
})

it.each(['markdown', 'generated'])('%s dimensions follow owner and source, not stale reads or hints', async kind => {
  const path = `${paths.second}?${kind}`

  const view = (connectionId: string, profile: string, src = path) => (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId, profile }}>
      {content(kind, src, 'portrait')}
    </ComposerScopeProvider>
  )

  const first = render(view('owner-a', 'a'))
  await decode(first.container, 900, 600)
  first.unmount()

  const mounted = render(view('owner-b', 'a'))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).not.toBe(1.5)
  mounted.rerender(view('owner-a', 'b'))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).not.toBe(1.5)
  mounted.rerender(view('owner-a', 'a'))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBe(1.5)
  const reserved = frame(mounted.container).style.cssText
  // Settle the two superseded owners after A has become visible again.
  await act(async () => {
    reads.shift()!.resolve(data)
    reads.shift()!.reject(new Error('retired'))
  })
  expect(frame(mounted.container).style.cssText).toBe(reserved)
  expect(mounted.container.querySelector('img')).toBeNull()
  // Same path, changed bytes: update the next mount's intrinsic dimensions,
  // but never reshape the frame currently being read.
  await decode(mounted.container, 600, 900)
  expect(frame(mounted.container).style.cssText).toBe(reserved)
  mounted.rerender(view('owner-a', 'a', `${path}-revision`))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).not.toBe(1.5)
  expect(mounted.container.querySelector('img')).toBeNull()
  await act(async () => reads.shift()!.reject(new Error('missing')))
  expect(mounted.container.textContent).toMatch(/Open image/i)
  mounted.rerender(view('owner-a', 'a'))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBeCloseTo(600 / 900)
  expect(window.hermesDesktop.api).toHaveBeenLastCalledWith({
    connectionId: 'owner-a',
    profile: 'a',
    path: `/api/fs/read-data-url?path=${encodeURIComponent(path)}`
  })
})
