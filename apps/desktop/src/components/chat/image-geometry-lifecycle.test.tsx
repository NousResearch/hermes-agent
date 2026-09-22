import { act, cleanup, fireEvent, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useMediaImage } from '@/hooks/use-media-image'
import { mediaImageKey, rememberMediaImageDimensions } from '@/lib/media'
import { $connection } from '@/store/session'

import { GeneratedImage } from './generated-image-result'

vi.mock('./image-generation-placeholder', () => ({ DiffusionCanvas: () => null }))

const data = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600"/>'
let reads: Array<(value: string) => void>
const read = vi.fn(() => new Promise<string>(resolve => reads.push(resolve)))

beforeEach(() => {
  reads = []
  read.mockClear()
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: read, api: vi.fn(() => read().then(dataUrl => ({ dataUrl }))) })
  $connection.set({ connectionId: 'lifecycle-local', mode: 'local', profile: 'a' } as never)
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.unstubAllGlobals()
})

function frame(container: HTMLElement) {
  return container.querySelector<HTMLElement>('[data-slot="aui_generated-image"]')!
}

async function decode(container: HTMLElement, width: number, height: number) {
  await act(async () => reads.shift()!(data))
  const img = container.querySelector('img')!
  Object.defineProperties(img, { naturalWidth: { value: width }, naturalHeight: { value: height } })
  fireEvent.load(img)
}

it.each([
  { name: 'result metadata', width: 900, height: 600, cached: false },
  { name: 'warm cache', width: 800, height: 800, cached: true },
  { name: 'same aspect but smaller intrinsic width', width: 160, height: 90, cached: false }
])('keeps the pending frame when the result arrives with $name', async ({ name, width, height, cached }) => {
  const path = `/lifecycle/${name}.svg`
  const result = { success: true, image: path, pixel_size: cached ? undefined : `${width}x${height}` }

  if (cached) {
    rememberMediaImageDimensions(mediaImageKey(path, $connection.get()), width, height)
  }

  const mounted = render(<GeneratedImage aspectRatio="landscape" />)
  const pendingStyle = frame(mounted.container).style.cssText
  mounted.rerender(<GeneratedImage aspectRatio="landscape" result={result} />)

  // This must hold before the file read settles, not just after img.onload.
  expect(frame(mounted.container).style.cssText).toBe(pendingStyle)
  expect(mounted.container.querySelector('img')).toBeNull()
  await decode(mounted.container, width, height)
  expect(frame(mounted.container).style.cssText).toBe(pendingStyle)
  mounted.unmount()

  // A fresh mount of the completed result still benefits from intrinsic size.
  const warm = render(<GeneratedImage aspectRatio="landscape" result={result} />)
  expect(parseFloat(frame(warm.container).style.aspectRatio)).toBeCloseTo(width / height)
  const warmStyle = frame(warm.container).style.cssText
  await decode(warm.container, width, height)
  fireEvent.error(warm.container.querySelector('img')!)
  expect(frame(warm.container).style.cssText).toBe(warmStyle)
})

it.each([
  { connectionId: 'owner-b', profile: 'a' },
  { connectionId: 'owner-a', profile: 'b' }
])('does not inherit a pending frame across owner $connectionId/$profile', owner => {
  const view = (connectionId: string, profile: string, result?: unknown) => (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId, profile }}>
      <GeneratedImage aspectRatio="landscape" result={result} />
    </ComposerScopeProvider>
  )

  const mounted = render(view('owner-a', 'a'))
  mounted.rerender(view(owner.connectionId, owner.profile, { image: '/lifecycle/other.svg', pixel_size: '600x900' }))
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBeCloseTo(600 / 900)
})

it('still resets the frame between two actual image sources', () => {
  const mounted = render(<GeneratedImage result={{ image: '/lifecycle/first.svg', pixel_size: '900x600' }} />)
  mounted.rerender(<GeneratedImage result={{ image: '/lifecycle/second.svg', pixel_size: '600x900' }} />)
  expect(parseFloat(frame(mounted.container).style.aspectRatio)).toBeCloseTo(600 / 900)
})

it('does not reserve an empty markdown image unless the caller opts in', () => {
  const mounted = renderHook(
    ({ path }) => useMediaImage(path, 16 / 9, path ? { width: 600, height: 900 } : undefined),
    {
      initialProps: { path: '' }
    }
  )

  expect(mounted.result.current.frameStyle.aspectRatio).toBeCloseTo(16 / 9)
  // Markdown callers keep the default source-based lifetime.
  mounted.rerender({ path: '/lifecycle/markdown.svg' })
  expect(mounted.result.current.frameStyle.aspectRatio).toBeCloseTo(600 / 900)
})
