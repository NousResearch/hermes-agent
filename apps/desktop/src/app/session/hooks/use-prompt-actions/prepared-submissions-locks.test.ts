import { webcrypto } from 'node:crypto'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import type { SubmissionDestination } from './submission-destination'

const originalLocks = Object.getOwnPropertyDescriptor(navigator, 'locks')

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  vi.stubGlobal('crypto', webcrypto)
  vi.stubGlobal('hermesDesktop', undefined)
})

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()

  if (originalLocks) { Object.defineProperty(navigator, 'locks', originalLocks) }
  else { Reflect.deleteProperty(navigator, 'locks') }
})

// WebKit rejects names longer than 1024 UTF-16 code units (issue #151619).
function boundedLocks() {
  const held = new Set<string>()

  const request = vi.fn((name: string, _options: LockOptions, callback: LockGrantedCallback<unknown>) => {
    if (name.length > 1024) { return Promise.reject(new DOMException('Lock name too long', 'NotSupportedError')) }

    if (held.has(name)) { return Promise.resolve(callback(null)) }

    held.add(name)

    return Promise.resolve(callback({ name, mode: 'exclusive' } as Lock)).finally(() => held.delete(name))
  })

  Object.defineProperty(navigator, 'locks', { configurable: true, value: { request } })

  return request
}

test('large prepared sends keep distinct, bounded locks and cannot be adopted from another live tab', async () => {
  const request = boundedLocks()
  const first = await import('./prepared-submissions')
  vi.resetModules()
  const second = await import('./prepared-submissions')
  const destination = { scopeKey: 'webapp::default' } as SubmissionDestination
  const text = 'a long pasted paragraph '.repeat(200)
  const intent = first.preparedSubmissionKey('stored', destination, text, [], { submission_id: 'first' })
  const slot = await first.preparedSubmissionSlot(intent)
  await first.writePreparedSubmission(slot, { id: 'first', owner: undefined, text, attachments: [], params: {} })

  expect(await second.adoptPreparedSubmission(intent, 'first')).toBeUndefined()
  const different = second.preparedSubmissionKey('stored', destination, text + 'different suffix', [], { submission_id: 'first' })
  const otherSlot = await second.preparedSubmissionSlot(different)
  expect(request.mock.calls.every(([name]) => name.length <= 1024)).toBe(true)
  expect(request.mock.calls[0][0]).toBe(request.mock.calls[1][0])
  expect(request.mock.calls[0][0]).not.toBe(request.mock.calls[2][0])

  await first.removePreparedSubmission(slot, 'first')
  await second.removePreparedSubmission(otherSlot, 'second')
})

test('a refused Web Lock rejects preparation instead of leaving the submit pending', async () => {
  const failure = new DOMException('Lock access refused', 'SecurityError')
  Object.defineProperty(navigator, 'locks', { configurable: true, value: { request: () => Promise.reject(failure) } })
  const prepared = await import('./prepared-submissions')
  await expect(prepared.preparedSubmissionSlot('["webapp::default","stored","short"]')).rejects.toBe(failure)
})
