// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchCatalog } from './catalog-data'

beforeEach(() => {
  vi.useFakeTimers()
  // Node's native timeout uses internal timers; keep the baseline's browser
  // timeout on the same deterministic clock as the response fixture.
  vi.spyOn(AbortSignal, 'timeout').mockImplementation(ms => {
    const controller = new AbortController()
    setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), ms)
    return controller.signal
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function serveStream() {
  let writer!: ReadableStreamDefaultController<Uint8Array>
  let signal!: AbortSignal
  const encoder = new TextEncoder()
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url, options) => {
      signal = options.signal
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            writer = controller
            signal.addEventListener('abort', () =>
              controller.error(new DOMException('The user aborted a request.', 'AbortError'))
            )
          }
        })
      )
    })
  )
  return {
    send: (text: string) => writer.enqueue(encoder.encode(text)),
    close: () => writer.close(),
    fail: (error: Error) => writer.error(error),
    signal: () => signal
  }
}

describe('catalog download liveness', () => {
  it.each(['skills', 'plugins'] as const)(
    'finishes a progressing %s download beyond the old total deadline',
    async kind => {
      const stream = serveStream()
      const result = fetchCatalog(kind).then(
        value => ({ value }),
        error => ({ error })
      )
      await vi.advanceTimersByTimeAsync(40_000)
      stream.send('[{"name":"slow-')
      await vi.advanceTimersByTimeAsync(40_000)
      expect(stream.signal().aborted).toBe(false)
      stream.send('catalog","source":"official"}]')
      stream.close()
      expect(await result).toMatchObject({ value: [{ name: 'slow-catalog' }] })
      expect(vi.getTimerCount()).toBe(0)
    }
  )

  it('reports a stalled body as a timeout, not user cancellation', async () => {
    const stream = serveStream()
    const result = fetchCatalog('skills').catch(error => error)
    await vi.advanceTimersByTimeAsync(40_000)
    stream.send('[')
    await vi.advanceTimersByTimeAsync(60_001)
    expect(stream.signal().aborted).toBe(true)
    expect((await result).message).toMatch(/timed out/i)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('bounds waiting for response headers', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url, options) =>
          new Promise((_resolve, reject) => {
            options.signal.addEventListener('abort', () =>
              reject(new DOMException('The user aborted a request.', 'AbortError'))
            )
          })
      )
    )
    const result = fetchCatalog('skills').catch(error => error)
    await vi.advanceTimersByTimeAsync(60_001)
    expect((await result).message).toMatch(/timed out/i)
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each(['[]', 'not json'])('cleans its deadline after completion or malformed JSON: %s', async body => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(body))
    )
    const result = await fetchCatalog('skills').catch(error => error)
    if (body === '[]') expect(result).toEqual([])
    else expect(result).toBeInstanceOf(SyntaxError)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('preserves transport errors and releases its deadline', async () => {
    const stream = serveStream()
    const error = new Error('connection reset')
    const result = fetchCatalog('skills').catch(error => error)
    await vi.advanceTimersByTimeAsync(1)
    stream.fail(error)
    expect(await result).toBe(error)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('preserves HTTP errors and releases its deadline', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('', { status: 503 }))
    )
    await expect(fetchCatalog('plugins')).rejects.toThrow('Catalog HTTP 503')
    expect(vi.getTimerCount()).toBe(0)
  })
})
