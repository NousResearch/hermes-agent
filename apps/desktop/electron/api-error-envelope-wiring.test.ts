/**
 * Wiring contract for the `hermes:api` 404 envelope.
 *
 * api-error-envelope.test.ts proves the envelope's shape; this file exercises
 * the two functions the ENDS actually call — `settleApiRequest` in
 * `ipcMain.handle('hermes:api')` and `unwrapApiResult` in preload's
 * `invokeHermesApi` — so the test measures shipped behaviour rather than a copy
 * of the handler body.
 *
 * What the bug was: Electron logs every rejected ipcMain.handle promise with a
 * main-process stack trace, and a REST 404 on this channel is control flow (a
 * fresh draft probed by id, a discarded draft's id still in renderer storage).
 * So the log filled with stack traces for a case the renderer already handles.
 */
import { describe, expect, it } from 'vitest'

import { isApiErrorEnvelope, settleApiRequest, unwrapApiResult } from './api-error-envelope'

// The renderer's real 404 predicate (src/lib/gateway-rpc.ts::isMissingRestEndpoint)
// lives outside this tsconfig project, and no electron test imports across that
// boundary. Mirror only the message pattern it keys on, so this file asserts the
// property that matters — the message still reads as a 404 after the round trip —
// without widening the project's file list.
const READS_AS_404 = /(?:^\s*|error:\s*)404\b/i

function httpError(statusCode: number, body: string): Error & { statusCode: number } {
  const error = new Error(`${statusCode}: ${body}`) as Error & { statusCode: number }
  error.statusCode = statusCode

  return error
}

const sessionNotFound = () => Promise.reject(httpError(404, '{"detail":"Session not found"}'))

/** The full round trip: main settles, preload unwraps. */
const roundTrip = <T>(dispatch: () => Promise<T>) => unwrapApiResult(() => settleApiRequest(dispatch))

describe('settleApiRequest (main-process side)', () => {
  it('RESOLVES a 404 so Electron has no rejected handler promise to log', async () => {
    // The assertion that encodes the bug: a rejection here is exactly what
    // Electron reported as "Error occurred in handler for 'hermes:api'".
    const result = await settleApiRequest(sessionNotFound)

    expect(isApiErrorEnvelope(result)).toBe(true)
  })

  it('still rejects a 500, so genuine faults stay visible in the log', async () => {
    await expect(settleApiRequest(() => Promise.reject(httpError(500, 'boom')))).rejects.toThrow('500: boom')
  })

  it('still rejects an auth failure and a transport failure', async () => {
    await expect(settleApiRequest(() => Promise.reject(httpError(401, 'nope')))).rejects.toThrow('401: nope')
    await expect(settleApiRequest(() => Promise.reject(new Error('socket hang up')))).rejects.toThrow('socket hang up')
  })

  it('passes a successful payload through untouched', async () => {
    const payload = { id: '20260830_120000_abcdef', title: 'a session' }

    await expect(settleApiRequest(() => Promise.resolve(payload))).resolves.toEqual(payload)
  })
})

describe('unwrapApiResult (preload side)', () => {
  it('rethrows the 404 so a fail-closed caller still sees a rejection', async () => {
    // resolveStoredSession relies on this: an envelope leaking through as a
    // RESOLVED object would make a missing session look like a real row.
    await expect(roundTrip(sessionNotFound)).rejects.toThrow('404: {"detail":"Session not found"}')
  })

  it('keeps message-based 404 detection working end to end', async () => {
    const error = await roundTrip(sessionNotFound).catch((err: unknown) => err)

    expect(READS_AS_404.test((error as Error).message)).toBe(true)
    expect((error as Error & { statusCode?: number }).statusCode).toBe(404)
  })

  it('does not swallow a payload that merely looks like an error', async () => {
    const payload = { message: '404: not really', statusCode: 404 }

    await expect(roundTrip(() => Promise.resolve(payload))).resolves.toEqual(payload)
  })

  it('leaves non-enveloped rejections reaching the renderer unchanged', async () => {
    await expect(roundTrip(() => Promise.reject(httpError(503, 'unavailable')))).rejects.toThrow('503: unavailable')
  })
})
