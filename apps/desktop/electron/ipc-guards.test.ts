import { describe, expect, it } from 'vitest'

import { cloneableError, replyAlways, withTimeout } from './ipc-guards'

// A rejection Electron cannot structured-clone: the exact shape that used to
// surface renderer-side as the opaque "reply was never sent".
function uncloneableRejection(): Promise<never> {
  const weird = Object.assign(new Error('weird'), { release: () => undefined, handle: process.stdout })
  return Promise.reject(weird)
}

describe('withTimeout', () => {
  it('passes a fast work value straight through', async () => {
    await expect(withTimeout(Promise.resolve('ok'), 1000, 'never fires')).resolves.toBe('ok')
  })

  it('rejects with a legible message when the work outlives the cap', async () => {
    await expect(withTimeout(new Promise<never>(() => undefined), 30, 'Timed out downloading the image')).rejects.toThrow(
      /Timed out downloading the image \(timed out after 0s\)/
    )
  })

  it('does not leave the losing work as an unhandled rejection', async () => {
    const lateFailure = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('late')), 50))

    await expect(withTimeout(lateFailure, 10, 'capped')).rejects.toThrow(/capped \(timed out after 0s\)/)

    // Let the losing rejection fire; an unhandled rejection here would crash
    // the vitest process.
    await new Promise(resolve => setTimeout(resolve, 80))
  })
})

describe('replyAlways — saveImageFromUrl contract (rethrow cloneable)', () => {
  it('converts a non-serializable rejection into a cloneable Error carrying the real cause', async () => {
    const outcome = replyAlways(() => uncloneableRejection(), message => {
      throw cloneableError(message)
    })

    await expect(outcome).rejects.toThrow('weird')
    await outcome.catch(error => {
      // The rethrown rejection must cross the structured-clone boundary.
      expect(() => JSON.parse(JSON.stringify({ message: error.message }))).not.toThrow()
      expect(error.message).toBe('weird')
    })
  })

  it('passes a successful save through untouched', async () => {
    await expect(replyAlways(() => Promise.resolve(true), message => { throw cloneableError(message) })).resolves.toBe(true)
  })
})

describe('replyAlways — fetchLinkTitle / resolveFavicon contract (degrade to empty)', () => {
  it('resolves empty on a non-serializable rejection instead of looping observe-only or rejecting', async () => {
    await expect(
      replyAlways(() => uncloneableRejection(), () => '', { timeoutMessage: 'Timed out fetching the link title', timeoutMs: 1000 })
    ).resolves.toBe('')
  })

  it('resolves empty on a timeout instead of hanging the void-style caller', async () => {
    await expect(
      replyAlways(() => new Promise<never>(() => undefined), () => '', {
        timeoutMessage: 'Timed out resolving the favicon',
        timeoutMs: 20
      })
    ).resolves.toBe('')
  })

  it('passes a real title/favicon through untouched', async () => {
    await expect(replyAlways(() => Promise.resolve('Some Title'), () => '')).resolves.toBe('Some Title')
  })
})
