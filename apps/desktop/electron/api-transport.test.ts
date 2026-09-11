/**
 * Unit + live-transport tests for the Electron main process's Hermes REST
 * retry policy (#92976 / PR #92977 salvage).
 *
 * The live tests run REAL node http servers that misbehave the way the
 * reported backend does (closing sockets under burst keep-alive traffic) and
 * prove two things end to end:
 *
 *  - idempotent GETs that die with ECONNRESET are retried and succeed, where
 *    a single bare attempt (pre-PR behavior) surfaces the raw reset;
 *  - a POST whose socket is reset AFTER the server processed it is NOT
 *    retried: the server-side hit counter stays at 1 and the error surfaces.
 */
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import { gzipSync } from 'node:zlib'

import { afterAll, describe, expect, it } from 'vitest'

import {
  decodeJsonResponseBody,
  destroyKeepaliveAgents,
  downloadAgentFor,
  isIdempotentMethod,
  isTransientTransportError,
  JSON_ACCEPT_ENCODING,
  jsonAgentFor,
  shouldRetryRequest,
  withRetry
} from './api-transport'

async function* bodyChunks(...chunks: Buffer[]) {
  yield* chunks
}

describe('bounded JSON response decoding', () => {
  it('advertises only the supported gzip encoding', () => {
    expect(JSON_ACCEPT_ENCODING).toBe('gzip')
  })

  it('reads identity responses without changing their bytes', async () => {
    const result = await decodeJsonResponseBody(bodyChunks(Buffer.from('{"ok":'), Buffer.from('true}')), undefined)

    expect(result.toString('utf8')).toBe('{"ok":true}')
  })

  it('decodes a gzip response', async () => {
    const compressed = gzipSync(Buffer.from('{"sessions":[1,2,3]}'))
    const result = await decodeJsonResponseBody(bodyChunks(compressed), 'GZip')

    expect(result.toString('utf8')).toBe('{"sessions":[1,2,3]}')
  })

  it('rejects unsupported response encodings instead of parsing compressed bytes', async () => {
    await expect(decodeJsonResponseBody(bodyChunks(Buffer.from('data')), 'br')).rejects.toThrow(
      'Unsupported JSON response content-encoding: br'
    )
  })

  it('rejects malformed gzip data', async () => {
    await expect(decodeJsonResponseBody(bodyChunks(Buffer.from('not gzip')), 'gzip')).rejects.toThrow()
  })

  it('bounds compressed bytes before buffering the full response', async () => {
    await expect(
      decodeJsonResponseBody(bodyChunks(Buffer.alloc(5), Buffer.alloc(5)), 'identity', {
        maxWireBytes: 8,
        maxDecodedBytes: 64
      })
    ).rejects.toThrow('JSON response exceeds the 8-byte wire limit')
  })

  it('bounds decompressed output to resist gzip bombs', async () => {
    const compressed = gzipSync(Buffer.alloc(1024, 65))

    await expect(
      decodeJsonResponseBody(bodyChunks(compressed), 'gzip', {
        maxWireBytes: 1024,
        maxDecodedBytes: 128
      })
    ).rejects.toThrow()
  })

  it('rejects trailing garbage after a valid gzip member', async () => {
    const compressed = Buffer.concat([gzipSync(Buffer.from('{"ok":true}')), Buffer.from('garbage')])

    await expect(decodeJsonResponseBody(bodyChunks(compressed), 'gzip')).rejects.toThrow()
  })
})

function errWithCode(code: string, message = code): NodeJS.ErrnoException {
  const e: NodeJS.ErrnoException = new Error(message)
  e.code = code

  return e
}

afterAll(() => {
  destroyKeepaliveAgents()
})

describe('isTransientTransportError', () => {
  it('accepts transient socket-level codes and messages', () => {
    for (const code of ['ECONNRESET', 'ECONNREFUSED', 'EPIPE', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN']) {
      expect(isTransientTransportError(errWithCode(code))).toBe(true)
    }

    expect(isTransientTransportError(new Error('socket hang up'))).toBe(true)
    expect(isTransientTransportError(new Error('read ECONNRESET'))).toBe(true)
  })

  it('rejects non-transport errors', () => {
    expect(isTransientTransportError(new Error('404: not found'))).toBe(false)
    expect(isTransientTransportError(new Error('Invalid JSON from http://x'))).toBe(false)
    expect(isTransientTransportError(null)).toBe(false)
    expect(isTransientTransportError(undefined)).toBe(false)
  })
})

describe('isIdempotentMethod', () => {
  it.each([
    ['GET', true],
    ['get', true],
    ['HEAD', true],
    ['OPTIONS', true],
    ['POST', false],
    ['PUT', false],
    ['PATCH', false],
    ['DELETE', false],
    [undefined, true] // node http defaults omitted method to GET
  ])('%s -> %s', (method, expected) => {
    expect(isIdempotentMethod(method)).toBe(expected)
  })
})

describe('shouldRetryRequest truth table', () => {
  const reset = () => errWithCode('ECONNRESET', 'read ECONNRESET')
  const refused = () => errWithCode('ECONNREFUSED', 'connect ECONNREFUSED 127.0.0.1:1')
  const hangUp = () => new Error('socket hang up')

  it('GET: retries any transient error regardless of body state', () => {
    expect(shouldRetryRequest(reset(), 'GET', { bodySent: true })).toBe(true)
    expect(shouldRetryRequest(reset(), 'GET', { bodySent: false })).toBe(true)
    expect(shouldRetryRequest(hangUp(), 'HEAD', { bodySent: true })).toBe(true)
  })

  it('GET: never retries non-transport errors (HTTP 4xx/5xx surfaced as Error)', () => {
    expect(shouldRetryRequest(new Error('500: boom'), 'GET', { bodySent: true })).toBe(false)
  })

  it('POST: retries when the connection provably never happened', () => {
    expect(shouldRetryRequest(refused(), 'POST', { bodySent: false })).toBe(true)
    expect(shouldRetryRequest(refused(), 'POST', { bodySent: true })).toBe(true) // refused == nothing sent
    expect(shouldRetryRequest(errWithCode('ENOTFOUND'), 'PUT', { bodySent: false })).toBe(true)
  })

  it('POST: retries transient errors thrown before the body was flushed', () => {
    expect(shouldRetryRequest(reset(), 'POST', { bodySent: false })).toBe(true)
    expect(shouldRetryRequest(hangUp(), 'DELETE', { bodySent: false })).toBe(true)
  })

  it('POST: does NOT retry ambiguous resets after the body went out', () => {
    expect(shouldRetryRequest(reset(), 'POST', { bodySent: true })).toBe(false)
    expect(shouldRetryRequest(hangUp(), 'POST', { bodySent: true })).toBe(false)
    expect(shouldRetryRequest(errWithCode('EPIPE'), 'PUT', { bodySent: true })).toBe(false)
    expect(shouldRetryRequest(errWithCode('ETIMEDOUT'), 'DELETE', { bodySent: true })).toBe(false)
  })

  it('POST: conservative when request state is unknown', () => {
    // No bodySent flag at all — treat as "may have been sent", don't retry.
    expect(shouldRetryRequest(reset(), 'POST', {})).toBe(false)
    expect(shouldRetryRequest(reset(), 'POST')).toBe(false)
  })
})

describe('withRetry', () => {
  const noDelay = { delayFn: () => Promise.resolve() }

  it('retries a GET through transient failures and resolves', async () => {
    let attempts = 0

    const result = await withRetry(
      () => {
        attempts += 1

        if (attempts < 3) {
          return Promise.reject(errWithCode('ECONNRESET'))
        }

        return Promise.resolve('ok')
      },
      { method: 'GET', ...noDelay }
    )

    expect(result).toBe('ok')
    expect(attempts).toBe(3)
  })

  it('gives each attempt a fresh requestState', async () => {
    const seen: boolean[] = []
    let attempts = 0
    await withRetry(
      (state: any) => {
        seen.push(state.bodySent)
        state.bodySent = true
        attempts += 1

        if (attempts < 2) {
          return Promise.reject(errWithCode('ECONNREFUSED'))
        }

        return Promise.resolve(null)
      },
      { method: 'POST', ...noDelay }
    )
    expect(seen).toEqual([false, false])
  })

  it('does not retry a POST that failed after the body was flushed', async () => {
    let attempts = 0
    await expect(
      withRetry(
        (state: any) => {
          attempts += 1
          state.bodySent = true

          return Promise.reject(errWithCode('ECONNRESET', 'read ECONNRESET'))
        },
        { method: 'POST', ...noDelay }
      )
    ).rejects.toThrow('read ECONNRESET')
    expect(attempts).toBe(1)
  })

  it('retries a POST on ECONNREFUSED (never reached the server)', async () => {
    let attempts = 0
    await expect(
      withRetry(
        () => {
          attempts += 1

          return Promise.reject(errWithCode('ECONNREFUSED'))
        },
        { method: 'POST', maxRetries: 2, ...noDelay }
      )
    ).rejects.toThrow('ECONNREFUSED')
    expect(attempts).toBe(3)
  })

  it('bounds retries at maxRetries even for GET', async () => {
    let attempts = 0
    await expect(
      withRetry(
        () => {
          attempts += 1

          return Promise.reject(errWithCode('ECONNRESET'))
        },
        { method: 'GET', maxRetries: 2, ...noDelay }
      )
    ).rejects.toThrow()
    expect(attempts).toBe(3)
  })

  it('never retries non-transient errors', async () => {
    let attempts = 0
    await expect(
      withRetry(
        () => {
          attempts += 1

          return Promise.reject(new Error('500: internal'))
        },
        { method: 'GET', ...noDelay }
      )
    ).rejects.toThrow('500')
    expect(attempts).toBe(1)
  })
})

describe('keep-alive agent pools', () => {
  it('separates JSON and download pools per protocol', () => {
    expect(jsonAgentFor('http:')).not.toBe(jsonAgentFor('https:'))
    expect(jsonAgentFor('http:')).not.toBe(downloadAgentFor('http:'))
    expect(jsonAgentFor('https:')).not.toBe(downloadAgentFor('https:'))
    // Stable across calls (a real pool, not a factory).
    expect(jsonAgentFor('http:')).toBe(jsonAgentFor('http:'))
  })
})

// ---------------------------------------------------------------------------
// LIVE transport tests against real misbehaving HTTP servers.
// ---------------------------------------------------------------------------

/** Minimal single-attempt GET mirroring the pre-PR fetchJson (no retry). */
function bareJsonGet(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.request(new URL(url), { agent: jsonAgentFor('http:'), method: 'GET' }, res => {
      const chunks: Buffer[] = []
      res.on('error', reject)
      res.on('data', c => chunks.push(c))
      res.on('end', () => resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))))
    })

    req.on('error', reject)
    req.end()
  })
}

/** The head behavior: same request under the verb-gated retry policy. */
function retriedJsonGet(url: string): Promise<any> {
  return withRetry(() => bareJsonGet(url), { method: 'GET', delayFn: () => Promise.resolve() })
}

function listen(server: http.Server): Promise<string> {
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => {
      resolve(`http://127.0.0.1:${(server.address() as AddressInfo).port}`)
    })
  })
}

describe('live: gzip JSON negotiation', () => {
  it('requests gzip and decodes the compressed response', async () => {
    const server = http.createServer((req, res) => {
      expect(req.headers['accept-encoding']).toBe(JSON_ACCEPT_ENCODING)
      res.setHeader('content-type', 'application/json')
      res.setHeader('content-encoding', 'gzip')
      res.end(gzipSync(Buffer.from('{"sessions":[{"id":"one"}]}')))
    })
    const base = await listen(server)

    try {
      const result = await new Promise<any>((resolve, reject) => {
        const req = http.request(
          new URL(`${base}/api/sessions`),
          {
            agent: jsonAgentFor('http:'),
            headers: { 'accept-encoding': JSON_ACCEPT_ENCODING }
          },
          res => {
            void decodeJsonResponseBody(res, res.headers['content-encoding'])
              .then(body => resolve(JSON.parse(body.toString('utf8'))))
              .catch(reject)
          }
        )

        req.on('error', reject)
        req.end()
      })

      expect(result).toEqual({ sessions: [{ id: 'one' }] })
    } finally {
      server.close()
    }
  })
})

describe('live: GET burst against a server that resets keep-alive sockets', () => {
  it('bare attempts fail with ECONNRESET/hang-up; retried GETs all succeed', async () => {
    // Deterministic misbehavior: every other request gets its socket
    // destroyed instead of a response — the observable client-side effect of
    // a backend killing idle keep-alive sockets mid-burst.
    let hits = 0

    const server = http.createServer((req, res) => {
      hits += 1

      if (hits % 2 === 1) {
        req.socket.destroy()

        return
      }

      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ n: hits }))
    })

    const base = await listen(server)

    try {
      // BASE (pre-PR, single attempt): the burst surfaces raw transport errors.
      let baseFailures = 0

      for (let i = 0; i < 6; i++) {
        try {
          await bareJsonGet(`${base}/api/sessions`)
        } catch (error: any) {
          baseFailures += 1
          expect(isTransientTransportError(error)).toBe(true)
        }
      }

      expect(baseFailures).toBeGreaterThan(0)

      // HEAD (retry policy): the same burst fully succeeds. Sequential so the
      // server's alternating destroy/respond pattern is deterministic per
      // request (first attempt reset, retry served).
      for (let i = 0; i < 6; i++) {
        const r = await retriedJsonGet(`${base}/api/sessions`)
        expect(r).toHaveProperty('n')
      }
    } finally {
      server.close()
    }
  }, 20_000)
})

describe('live: POST reset after server-side processing', () => {
  it('does not double-submit: server hit count stays 1, error surfaces', async () => {
    // The server fully receives and "processes" the POST (counter increments),
    // then RSTs the socket before responding — the dangerous ambiguous case.
    let posts = 0

    const server = http.createServer((req, res) => {
      const chunks: Buffer[] = []
      req.on('data', c => chunks.push(c))
      req.on('end', () => {
        posts += 1 // processed: prompt submitted / session created
        req.socket.resetAndDestroy()
        void res
      })
    })

    const base = await listen(server)

    const postOnce = () =>
      withRetry(
        (state: any) =>
          new Promise((resolve, reject) => {
            const body = Buffer.from(JSON.stringify({ prompt: 'hello' }))

            const req = http.request(
              new URL(`${base}/api/prompt`),
              {
                agent: jsonAgentFor('http:'),
                method: 'POST',
                headers: { 'content-type': 'application/json', 'content-length': String(body.length) }
              },
              res => {
                res.resume()
                res.on('end', () => resolve(null))
              }
            )

            req.on('error', reject)
            state.bodySent = true
            req.write(body)
            req.end()
          }),
        { method: 'POST', delayFn: () => Promise.resolve() }
      )

    try {
      await expect(postOnce()).rejects.toSatisfy((error: any) => isTransientTransportError(error))
      expect(posts).toBe(1) // exactly one server-side submission — no retry
    } finally {
      server.close()
    }
  }, 20_000)

  it('sanity: an identical GET-shaped retry WOULD have re-hit the server', async () => {
    // Companion proof that the verb gate (not luck) is what kept posts === 1:
    // the same reset-after-processing server sees multiple hits under GET.
    let gets = 0

    const server = http.createServer(req => {
      gets += 1
      req.socket.resetAndDestroy()
    })

    const base = await listen(server)

    try {
      await expect(retriedJsonGet(`${base}/api/thing`)).rejects.toThrow()
      expect(gets).toBeGreaterThan(1) // retried — proves the machinery fires
    } finally {
      server.close()
    }
  }, 20_000)
})
