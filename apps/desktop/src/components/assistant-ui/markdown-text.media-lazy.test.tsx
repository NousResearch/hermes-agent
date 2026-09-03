import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { mediaMarkdownHref } from '@/lib/media'
import { $connection } from '@/store/session'

import { MarkdownImage, MarkdownTextContent } from './markdown-text'

interface ObserverEntry {
  isIntersecting: boolean
  target: Element
}

class FakeIntersectionObserver {
  static instances: FakeIntersectionObserver[] = []

  readonly observed = new Set<Element>()

  constructor(
    private readonly callback: (entries: ObserverEntry[]) => void,
    readonly options?: IntersectionObserverInit
  ) {
    FakeIntersectionObserver.instances.push(this)
  }

  disconnect = vi.fn(() => this.observed.clear())
  observe = vi.fn((target: Element) => this.observed.add(target))
  unobserve = vi.fn((target: Element) => this.observed.delete(target))

  intersect(target: Element) {
    if (this.observed.has(target)) {
      this.callback([{ isIntersecting: true, target }])
    }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function admit(observer = FakeIntersectionObserver.instances.at(-1)) {
  expect(observer).toBeDefined()
  const target = [...observer!.observed][0]
  expect(target).toBeDefined()

  act(() => observer!.intersect(target))

  return observer!
}

describe('lazy transcript media', () => {
  const api = vi.fn(async (_request: { connectionId?: string; path: string; profile?: string }) => ({
    dataUrl: 'data:image/svg+xml;base64,PHN2Zy8+'
  }))

  const saveGatewayFile = vi.fn(async () => ({ saved: true }))
  const saveImageFromUrl = vi.fn(async () => true)
  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    FakeIntersectionObserver.instances = []
    api.mockReset()
    api.mockResolvedValue({ dataUrl: 'data:image/svg+xml;base64,PHN2Zy8+' })
    saveGatewayFile.mockClear()
    saveImageFromUrl.mockClear()
    originalDesktop = window.hermesDesktop
    vi.stubGlobal('IntersectionObserver', FakeIntersectionObserver)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api, saveGatewayFile, saveImageFromUrl }
    })
    $connection.set({ connectionId: 'gateway-a', mode: 'remote', profile: 'remote-work' } as never)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    vi.unstubAllGlobals()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('does not resolve an off-screen filesystem image', () => {
    render(<MarkdownImage alt="diagram" src="/tmp/offscreen-diagram.svg" />)

    expect(api).not.toHaveBeenCalled()
    expect(FakeIntersectionObserver.instances).toHaveLength(1)
    expect(FakeIntersectionObserver.instances[0].options?.rootMargin).toBe('600px 0px')
  })

  it('uses the transcript scrollport as the observer root', () => {
    const { container } = render(
      <div data-slot="aui_thread-viewport">
        <MarkdownImage alt="diagram" height={240} src="/tmp/scrollport.svg" width={320} />
      </div>
    )

    expect(FakeIntersectionObserver.instances[0].options?.root).toBe(container.firstElementChild)
  })

  it('keeps filesystem placeholders accessible and reserves their dimensions', () => {
    render(<MarkdownImage alt="A diagram" height={240} src="/tmp/placeholder.svg" width={320} />)

    const placeholder = screen.getByRole('img', { name: 'A diagram' })
    expect(placeholder.getAttribute('aria-busy')).toBe('true')
    expect(placeholder.getAttribute('width')).toBe('320')
    expect(placeholder.getAttribute('height')).toBe('240')
    expect(screen.getByText('Loading placeholder.svg...')).toBeTruthy()
  })

  it.each([
    ['relative', 'images/relative.svg'],
    ['blob', 'blob:https://example.test/image-id']
  ])('renders %s sources immediately without lazy filesystem gating', (_kind, src) => {
    render(<MarkdownImage alt="inline diagram" src={src} />)

    expect(screen.getByRole('img', { name: 'inline diagram' }).getAttribute('src')).toBe(src)
    expect(FakeIntersectionObserver.instances).toHaveLength(0)
    expect(api).not.toHaveBeenCalled()
  })

  it('does not eagerly resolve an off-screen MEDIA image link', async () => {
    const href = mediaMarkdownHref('/tmp/offscreen-media-link.svg')

    render(<MarkdownTextContent isRunning={false} text={`[Image: linked.svg](${href})`} />)

    await waitFor(() => expect(FakeIntersectionObserver.instances).toHaveLength(1))
    expect(api).not.toHaveBeenCalled()
  })

  it.each([
    ['web', 'https://example.test/diagram.svg'],
    ['data', 'data:image/svg+xml;base64,aW5saW5l']
  ])('renders an inline %s image immediately without observing or resolving it', (_kind, src) => {
    render(<MarkdownImage alt="inline diagram" src={src} />)

    const image = screen.getByRole('img', { name: 'inline diagram' })
    expect(image.getAttribute('src')).toBe(src)
    expect(api).not.toHaveBeenCalled()
    expect(FakeIntersectionObserver.instances).toHaveLength(0)
  })

  it('admits a near-viewport image only once', async () => {
    const view = render(<MarkdownImage alt="diagram" src="/tmp/admitted-once.svg" />)
    const observer = admit()

    await screen.findByRole('img', { name: 'diagram' })
    view.rerender(<MarkdownImage alt="diagram" src="/tmp/admitted-once.svg" />)

    expect(api).toHaveBeenCalledTimes(1)
    expect(observer.disconnect).toHaveBeenCalledTimes(1)
    expect(FakeIntersectionObserver.instances).toHaveLength(1)
  })

  it('sets native lazy-loading and async decoding on the admitted image', async () => {
    render(<MarkdownImage alt="diagram" decoding="sync" loading="eager" src="/tmp/native-lazy.svg" />)
    admit()

    const image = (await screen.findAllByRole('img', { name: 'diagram' })).at(-1)
    expect(image?.getAttribute('loading')).toBe('lazy')
    expect(image?.getAttribute('decoding')).toBe('async')
  })

  it('preserves image alt text, lightbox opening, and resolved-source download semantics', async () => {
    const resolvedSrc = 'data:image/svg+xml;base64,YWN0aW9ucw=='
    api.mockResolvedValue({ dataUrl: resolvedSrc })

    const { baseElement } = render(<MarkdownImage alt="Architecture diagram" src="/tmp/actions.svg" />)
    admit()

    const image = await screen.findByRole('img', { name: 'Architecture diagram' })
    expect(image.getAttribute('alt')).toBe('Architecture diagram')

    fireEvent.click(screen.getByRole('button', { name: 'Download image' }))
    await waitFor(() => expect(saveImageFromUrl).toHaveBeenCalledWith(resolvedSrc))

    fireEvent.click(screen.getByTitle('Open image'))
    await waitFor(() => expect(baseElement.querySelectorAll('img[alt="Architecture diagram"]')).toHaveLength(2))
  })

  it('deduplicates identical in-flight requests in the same connection scope', async () => {
    const request = deferred<{ dataUrl: string }>()
    api.mockReturnValue(request.promise)

    render(
      <>
        <MarkdownImage alt="first" src="/tmp/shared-request.svg" />
        <MarkdownImage alt="second" src="/tmp/shared-request.svg" />
      </>
    )

    admit(FakeIntersectionObserver.instances[0])
    admit(FakeIntersectionObserver.instances[1])

    expect(api).toHaveBeenCalledTimes(1)

    await act(async () => request.resolve({ dataUrl: 'data:image/svg+xml;base64,c2hhcmVk' }))
    await waitFor(() =>
      expect(screen.getByRole('img', { name: 'first' }).getAttribute('src')).toBe('data:image/svg+xml;base64,c2hhcmVk')
    )
    expect(screen.getByRole('img', { name: 'second' }).getAttribute('src')).toBe('data:image/svg+xml;base64,c2hhcmVk')
  })

  it('deduplicates only while a request is in flight', async () => {
    api
      .mockResolvedValueOnce({ dataUrl: 'data:image/svg+xml;base64,Zmlyc3Q=' })
      .mockResolvedValueOnce({ dataUrl: 'data:image/svg+xml;base64,c2Vjb25k' })

    const first = render(<MarkdownImage alt="first" src="/tmp/settled-request.svg" />)
    admit()

    await waitFor(() =>
      expect(screen.getByRole('img', { name: 'first' }).getAttribute('src')).toBe('data:image/svg+xml;base64,Zmlyc3Q=')
    )
    first.unmount()

    render(<MarkdownImage alt="second" src="/tmp/settled-request.svg" />)
    admit()

    await waitFor(() =>
      expect(screen.getByRole('img', { name: 'second' }).getAttribute('src')).toBe('data:image/svg+xml;base64,c2Vjb25k')
    )
    expect(api).toHaveBeenCalledTimes(2)
  })

  it('requires fresh viewport admission when a reused image receives a new path', async () => {
    const newRequest = deferred<{ dataUrl: string }>()
    api.mockImplementation(({ path }: { path: string }) =>
      path.includes('new.svg') ? newRequest.promise : Promise.resolve({ dataUrl: 'data:image/svg+xml;base64,b2xk' })
    )

    const view = render(<MarkdownImage alt="diagram" src="/tmp/old.svg" />)
    admit()
    await screen.findByRole('img', { name: 'diagram' })

    view.rerender(<MarkdownImage alt="diagram" src="/tmp/new.svg" />)
    await act(async () => undefined)

    expect(api).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Loading new.svg...')).not.toBeNull()
    expect(screen.getAllByRole('img', { name: 'diagram' })).toHaveLength(1)
    expect(FakeIntersectionObserver.instances).toHaveLength(2)

    admit()
    expect(api).toHaveBeenCalledTimes(2)

    await act(async () => newRequest.resolve({ dataUrl: 'data:image/svg+xml;base64,bmV3' }))
    expect((await screen.findByRole('img', { name: 'diagram' })).getAttribute('src')).toBe(
      'data:image/svg+xml;base64,bmV3'
    )
  })

  it('does not let a stale request overwrite a reused image node', async () => {
    const oldRequest = deferred<{ dataUrl: string }>()
    const newRequest = deferred<{ dataUrl: string }>()
    api.mockImplementation(({ path }: { path: string }) =>
      path.includes('old.svg') ? oldRequest.promise : newRequest.promise
    )

    const view = render(<MarkdownImage alt="diagram" src="/tmp/stale-old.svg" />)
    admit()
    view.rerender(<MarkdownImage alt="diagram" src="/tmp/stale-new.svg" />)
    admit()

    await act(async () => newRequest.resolve({ dataUrl: 'data:image/svg+xml;base64,bmV3' }))
    const image = await screen.findByRole('img', { name: 'diagram' })
    expect(image.getAttribute('src')).toBe('data:image/svg+xml;base64,bmV3')

    await act(async () => oldRequest.resolve({ dataUrl: 'data:image/svg+xml;base64,b2xk' }))
    expect(screen.getByRole('img', { name: 'diagram' }).getAttribute('src')).toBe(
      'data:image/svg+xml;base64,bmV3'
    )
  })

  it('does not publish an admitted result after unmount', async () => {
    const request = deferred<{ dataUrl: string }>()
    api.mockReturnValue(request.promise)

    const view = render(<MarkdownImage alt="diagram" src="/tmp/unmounted.svg" />)
    admit()
    view.unmount()

    await act(async () => request.resolve({ dataUrl: 'data:image/svg+xml;base64,dW5tb3VudGVk' }))

    expect(screen.queryByRole('img', { name: 'diagram' })).toBeNull()
  })

  it('does not reuse a pending result across connection and profile scopes', async () => {
    const firstRequest = deferred<{ dataUrl: string }>()
    const secondRequest = deferred<{ dataUrl: string }>()
    api.mockImplementation(({ profile }: { profile?: string }) =>
      profile === 'remote-work' ? firstRequest.promise : secondRequest.promise
    )

    render(<MarkdownImage alt="diagram" src="/tmp/scoped.svg" />)
    admit()

    act(() => {
      $connection.set({ connectionId: 'gateway-b', mode: 'remote', profile: 'personal' } as never)
    })

    await waitFor(() => expect(api).toHaveBeenCalledTimes(2))
    expect(api.mock.calls.map(([request]) => request.profile)).toEqual(['remote-work', 'personal'])

    await act(async () => firstRequest.resolve({ dataUrl: 'data:image/svg+xml;base64,c3RhbGU=' }))
    expect(screen.getByText('Loading scoped.svg...')).not.toBeNull()

    await act(async () => secondRequest.resolve({ dataUrl: 'data:image/svg+xml;base64,Y3VycmVudA==' }))
    expect((await screen.findByRole('img', { name: 'diagram' })).getAttribute('src')).toBe(
      'data:image/svg+xml;base64,Y3VycmVudA=='
    )
  })

  it('keeps the failed-image open fallback scoped to the active profile', async () => {
    api.mockRejectedValue(new Error('missing'))
    render(<MarkdownImage alt="diagram" src="/tmp/missing.svg" />)
    admit()

    const open = await screen.findByRole('button', { name: 'Open image' })
    fireEvent.click(open)

    await waitFor(() => {
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'gateway-a',
        path: '/tmp/missing.svg',
        profile: 'remote-work',
        suggestedName: 'missing.svg'
      })
    })
  })
})
