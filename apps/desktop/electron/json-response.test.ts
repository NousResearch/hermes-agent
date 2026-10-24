import http from 'node:http'
import type { AddressInfo } from 'node:net'
import { gzipSync } from 'node:zlib'

import { describe, expect, it, vi } from 'vitest'

import { htmlResponseError, httpStatusError } from './api-transport'
import { asksForGzip, decodeJsonResponse, decodeJsonResponseBody, requestJsonResponse } from './json-response'

describe('Node JSON response compression', () => {
  it('decodes gzip after concatenation and preserves other encodings', () => {
    const text = JSON.stringify({ message: 'transcript ✓' })
    const compressed = gzipSync(text)

    expect(decodeJsonResponseBody([compressed.subarray(0, 7), compressed.subarray(7)], 'gzip')).toBe(text)

    for (const encoding of [undefined, 'identity', 'br']) {
      expect(decodeJsonResponseBody([Buffer.from(text)], encoding)).toBe(text)
    }
  })

  it('passes an empty body marked gzip through, so the caller still sees its status', () => {
    expect(decodeJsonResponseBody([], 'gzip')).toBe('')
  })

  it('asks only remote backends for gzip', () => {
    for (const href of [
      'http://127.0.0.1:9119/',
      'http://127.0.0.2/',
      'http://localhost:9119/',
      'http://LOCALHOST/',
      'http://[::1]:9119/',
      'http://desktop.localhost/',
      'http://0.0.0.0:9119/',
      'http://[::]:9119/',
      'http://[::ffff:127.0.0.1]:9119/'
    ]) {
      expect(asksForGzip(new URL(href))).toBe(false)
      // An SSH backend's REST rides a 127.0.0.1 port forward; its descriptor marks it remote.
      expect(asksForGzip(new URL(href), true)).toBe(true)
    }

    for (const href of [
      'https://agent.example.com/',
      'http://100.64.0.7:9119/',
      'http://192.168.1.20:9119/',
      'http://[fd7a:115c::1]:9119/',
      'http://localhost.example.com/'
    ]) {
      expect(asksForGzip(new URL(href))).toBe(true)
    }
  })

  it('rejects corrupt gzip without throwing out of the response callback', () => {
    const reject = vi.fn()
    const onText = vi.fn()

    expect(() => decodeJsonResponse([Buffer.from('corrupt')], 'gzip', reject, onText)).not.toThrow()
    expect(reject).toHaveBeenCalledOnce()
    expect(onText).not.toHaveBeenCalled()
  })

  it('negotiates gzip over HTTP and delivers decoded JSON and error bodies', async () => {
    const requests: string[] = []

    const server = http.createServer((req, res) => {
      requests.push(String(req.headers['accept-encoding']))
      const path = req.url

      const body =
        path === '/html'
          ? '<!doctype html><html>login</html>'
          : path === '/invalid'
            ? 'not json'
            : path === '/error'
              ? 'backend unavailable'
              : JSON.stringify({ messages: ['transcript ✓'] })

      res.writeHead(path === '/error' ? 503 : 200, {
        'Content-Encoding': 'gzip',
        'Content-Type': path === '/html' ? 'text/html' : 'application/json'
      })
      res.end(gzipSync(body))
    })

    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const port = (server.address() as AddressInfo).port
    // A remote backend's URL, connected to this loopback server through the options' hostname.
    const remote = `http://backend.example:${port}`

    const read = (path: string, base = remote, remoteBackend = false) =>
      new Promise((resolve, reject) => {
        const url = new URL(path, base)

        const request = requestJsonResponse(
          http,
          url,
          { agent: false, hostname: '127.0.0.1' },
          res => {
            const chunks: Buffer[] = []

            res.on('error', reject)
            res.on('data', chunk => chunks.push(chunk))
            res.on('end', () =>
              decodeJsonResponse(chunks, res.headers['content-encoding'], reject, text => {
                if (res.statusCode! >= 400) {
                  reject(httpStatusError(res.statusCode, text, res.statusMessage))
                } else if (res.headers['content-type']?.includes('text/html')) {
                  reject(htmlResponseError(url.href, res.statusCode))
                } else {
                  resolve(JSON.parse(text))
                }
              })
            )
          },
          remoteBackend
        )

        request.on('error', reject)
        request.end()
      })

    try {
      await expect(read('/json')).resolves.toEqual({ messages: ['transcript ✓'] })
      await expect(read('/html')).rejects.toThrow(/HTML/)
      await expect(read('/invalid')).rejects.toThrow()
      await expect(read('/error')).rejects.toMatchObject({ statusCode: 503, message: '503: backend unavailable' })
      // A loopback backend is not asked for gzip, but a gzip body still decodes.
      const loopback = `http://127.0.0.1:${port}`

      await expect(read('/json', loopback)).resolves.toEqual({ messages: ['transcript ✓'] })
      // An SSH tunnel is a loopback URL to a remote backend: asked for gzip.
      await expect(read('/json', loopback, true)).resolves.toEqual({ messages: ['transcript ✓'] })
      expect(requests).toEqual(['gzip', 'gzip', 'gzip', 'gzip', 'undefined', 'gzip'])
    } finally {
      await new Promise<void>((resolve, reject) => server.close(error => (error ? reject(error) : resolve())))
    }
  })
})
