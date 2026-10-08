import { afterEach, expect, test, vi } from 'vitest'

import { listPreparedDrafts, type PreparedSubmission, preparedSubmissionKey, readPreparedSubmission, readRetryablePreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
import type { SubmissionDestination } from './submission-destination'

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

test('an uncertain legacy send explains its block and preserves its original identity', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const entry = { id: 'legacy', text: 'maybe delivered', attachments: [], params: {}, legacyAttempted: true } as unknown as PreparedSubmission
  await writePreparedSubmission('legacy', entry)
  await expect(readRetryablePreparedSubmission('legacy')).rejects.toThrow('acknowledgement was lost')
  expect(await readPreparedSubmission('legacy')).toEqual(entry)
})

test('separate identical sends retain independent journals and acknowledgement removes only its own send', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const first = preparedSubmissionKey('stored', destination, 'same text', [])
  const second = preparedSubmissionKey('stored', destination, 'same text', [])
  expect(second).not.toBe(first)
  const entry = (id: string): PreparedSubmission => ({ id, text: 'same text', attachments: [], params: { submission_id: id }, owner: { connectionId: 'local', profile: 'default' } })
  await Promise.all([writePreparedSubmission(first, entry('first')), writePreparedSubmission(second, entry('second'))])
  await removePreparedSubmission(first)
  expect(await readPreparedSubmission(second)).toMatchObject({ id: 'second' })
  const retry = { submission_id: 'retained-id' }
  expect(preparedSubmissionKey('stored', destination, 'same text', [], retry)).toBe(preparedSubmissionKey('stored', destination, 'same text', [], retry))
})

test.each([false, true])('explicit restored identity retrieves its original payload across display/key changes (legacy window slot: %s)', async legacySlot => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const key = preparedSubmissionKey('stored', destination, 'caption', [], { submission_id: 'one-operation', fromQueue: true })
  const original = legacySlot ? JSON.stringify([...JSON.parse(key), 'legacy-window-slot']) : key
  const restored = preparedSubmissionKey('stored', destination, 'caption\n', [], { submission_id: 'one-operation' })
  const entry = { id: 'one-operation', text: 'committed wire', attachments: [], params: { queued: true }, owner: { connectionId: 'local', profile: 'default' } } as PreparedSubmission
  await writePreparedSubmission(original, entry)
  expect(await readPreparedSubmission(restored)).toEqual(entry)
  await writePreparedSubmission(restored, entry)
  expect(Object.keys(JSON.parse(localStorage.getItem('hermes.desktop.preparedSubmissions.v1')!))).toEqual([original])
  await removePreparedSubmission(restored)
  expect(await readPreparedSubmission(original)).toBeUndefined()
})

test('recovery offers the original slash invocation and leaves its exact expanded journal intact', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const attachments: PreparedSubmission['attachments'] = [{ id: '/cache/a.png', occurrenceId: 'image-occurrence', kind: 'image', label: 'a.png' }]
  const ordinary = preparedSubmissionKey('stored', destination, 'caption', attachments)
  const slash = preparedSubmissionKey('stored', destination, 'expanded skill instructions', attachments, { retryText: '/skill task', submission_id: 'skill-id' })

  for (const [key, text] of [[ordinary, 'caption'], [slash, 'expanded skill instructions']]) {
    await writePreparedSubmission(key, { id: key, text, attachments, params: { submission_id: key }, owner: { connectionId: 'local', profile: 'default' } })
  }

  const before = localStorage.getItem('hermes.desktop.preparedSubmissions.v1')
  expect((await listPreparedDrafts('stored', destination.scopeKey)).map(entry => [entry.key, entry.text])).toEqual([[ordinary, 'caption'], [slash, '/skill task']])
  expect(await listPreparedDrafts('another', destination.scopeKey)).toEqual([])
  expect(localStorage.getItem('hermes.desktop.preparedSubmissions.v1')).toBe(before)
  expect((await readPreparedSubmission(slash))?.text).toBe('expanded skill instructions')
})

test('native preparation waits for acknowledgement and never downgrades a write failure to browser storage', async () => {
  let ack!: () => void
  const gate = new Promise<void>(resolve => { ack = resolve })
  const entry = { id: 'a', text: 'Ω\n  exact', attachments: [], params: { session_id: 'live' } } as unknown as PreparedSubmission
  const native = { read: vi.fn(async () => JSON.stringify({ key: entry })), update: vi.fn(() => gate) }
  vi.stubGlobal('hermesDesktop', { preparedSubmissions: native })
  let finished = false
  const writing = writePreparedSubmission('key', entry).then(() => { finished = true })
  await Promise.resolve()
  expect(finished).toBe(false)
  await vi.waitFor(() => expect(native.update).toHaveBeenCalledWith('key', JSON.stringify(entry)))
  ack(); await writing
  expect(await readPreparedSubmission('key')).toEqual(entry)
  native.update.mockRejectedValueOnce(new Error('disk full'))
  await expect(writePreparedSubmission('key', entry)).rejects.toThrow('disk full')
  expect(localStorage.length).toBe(0)
  await removePreparedSubmission('key')
  expect(native.update).toHaveBeenLastCalledWith('key', null)
  vi.stubGlobal('hermesDesktop', undefined)
  await writePreparedSubmission('key', entry)
  expect(await readPreparedSubmission('key')).toEqual(entry)
  await removePreparedSubmission('key')
  expect(await readPreparedSubmission('key')).toBeUndefined()
})
