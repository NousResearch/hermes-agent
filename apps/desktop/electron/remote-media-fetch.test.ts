import { describe, expect, it, vi } from 'vitest'

import { fetchRemoteMedia } from './remote-media-fetch'

describe('remote media response ownership', () => {
  it('aborts each cancelled stream, including an outstanding read, without cancelling siblings', async () => {
    const signals: AbortSignal[] = []
    const cancellations = Array.from({ length: 8 }, () => vi.fn())

    const responses = await Promise.all(
      cancellations.map(async cancel =>
        fetchRemoteMedia(async (_headers, signal) => {
          signals.push(signal)

          return new Response(new ReadableStream({ cancel }))
        }, new Headers())
      )
    )

    for (const [index, response] of responses.entries()) {
      const reader = response.body!.getReader()
      const pending = reader.read()
      await reader.cancel('player removed')
      await expect(pending).resolves.toEqual({ done: true, value: undefined })
      expect(signals[index].aborted).toBe(true)
      expect(signals[index].reason).toBe('player removed')
      expect(cancellations[index]).toHaveBeenCalledWith('player removed')
      expect(signals.slice(index + 1).every(signal => !signal.aborted)).toBe(true)
    }
  })

  it('preserves full/range/error/empty responses and releases signal listeners on every terminal path', async () => {
    for (const status of [200, 206, 416]) {
      const request = new AbortController()
      const remove = vi.spyOn(request.signal, 'removeEventListener')
      const headers = new Headers({ range: 'bytes=4-7', 'if-range': 'etag' })
      const bytes = new Uint8Array([0, 255, 42, 7])

      const response = await fetchRemoteMedia(
        async (forwarded, signal) => {
          expect(forwarded).toBe(headers)
          expect(signal.aborted).toBe(false)

          return new Response(bytes, {
            status,
            statusText: 'media',
            headers: { 'content-range': 'bytes 4-7/8', etag: 'etag' }
          })
        },
        headers,
        request.signal
      )

      expect(response.status).toBe(status)
      expect(response.statusText).toBe('media')
      expect(response.headers.get('content-range')).toBe('bytes 4-7/8')
      expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes)
      expect(remove).toHaveBeenCalledOnce()
    }

    for (const status of [200, 204, 304]) {
      const request = new AbortController()
      const remove = vi.spyOn(request.signal, 'removeEventListener')
      const upstream = new Response(null, { status })
      expect(await fetchRemoteMedia(async () => upstream, new Headers(), request.signal)).toBe(upstream)
      expect(remove).toHaveBeenCalledOnce()
    }

    const request = new AbortController()
    let upstreamSignal: AbortSignal | undefined

    const response = await fetchRemoteMedia(
      async (_headers, signal) => {
        upstreamSignal = signal

        return new Response(
          new ReadableStream({
            start(target) {
              signal.addEventListener('abort', () => target.error(signal.reason), { once: true })
            }
          })
        )
      },
      new Headers(),
      request.signal
    )

    request.abort(new Error('request cancelled'))
    await expect(response.arrayBuffer()).rejects.toThrow('request cancelled')
    expect(upstreamSignal?.aborted).toBe(true)
    const neverFetch = vi.fn()
    await expect(fetchRemoteMedia(neverFetch, new Headers(), request.signal)).rejects.toThrow('request cancelled')
    expect(neverFetch).not.toHaveBeenCalled()
    const failure = new Error('upstream failed')
    const failedRequest = new AbortController()
    const remove = vi.spyOn(failedRequest.signal, 'removeEventListener')
    await expect(
      fetchRemoteMedia(
        async () => {
          throw failure
        },
        new Headers(),
        failedRequest.signal
      )
    ).rejects.toBe(failure)
    expect(remove).toHaveBeenCalledOnce()
  })
})
