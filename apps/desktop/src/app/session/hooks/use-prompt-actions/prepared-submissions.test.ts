import { afterEach, expect, test, vi } from 'vitest'

import { listPreparedImageDrafts, type PreparedSubmission, preparedSubmissionKey, readPreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
import { storedJournal } from './prepared-submissions-test-utils'
import type { SubmissionDestination } from './submission-destination'

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

test('image recovery never offers expanded slash intent as an ordinary draft and leaves its exact journal intact', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const attachments: PreparedSubmission['attachments'] = [{ id: '/cache/a.png', occurrenceId: 'image-occurrence', kind: 'image', label: 'a.png' }]
  const ordinary = preparedSubmissionKey('stored', destination, 'caption', attachments)
  const slash = preparedSubmissionKey('stored', destination, 'expanded skill instructions', attachments, { retryText: '/skill task', submission_id: 'skill-id' })

  for (const [key, text] of [[ordinary, 'caption'], [slash, 'expanded skill instructions']]) {
    await writePreparedSubmission(key, { id: key, text, attachments, params: { submission_id: key }, owner: { connectionId: 'local', profile: 'default' } })
  }

  const before = storedJournal()
  expect((await listPreparedImageDrafts('stored', destination.scopeKey)).map(entry => entry.key)).toEqual([ordinary])
  expect(await listPreparedImageDrafts('another', destination.scopeKey)).toEqual([])
  expect(storedJournal()).toEqual(before)
  expect((await readPreparedSubmission(slash))?.text).toBe('expanded skill instructions')
})

test('a browser journal read lists the storage keys once, however many stores it reads', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const owner = { connectionId: 'local', profile: 'default' }

  for (const id of ['a', 'b', 'c']) {
    await writePreparedSubmission(id, { id, text: id, attachments: [], params: {}, owner })
  }

  await removePreparedSubmission('c', 'c')
  const key = vi.spyOn(window.localStorage, 'key')
  expect((await readPreparedSubmission('a'))?.text).toBe('a')
  expect(await readPreparedSubmission('c')).toBeUndefined()
  expect(key).toHaveBeenCalledTimes(2 * window.localStorage.length)
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
  expect(native.update).toHaveBeenCalledWith('key', JSON.stringify(entry))
  ack(); await writing
  expect(await readPreparedSubmission('key')).toEqual(entry)
  native.update.mockRejectedValueOnce(new Error('disk full'))
  await expect(writePreparedSubmission('key', entry)).rejects.toThrow('disk full')
  expect(localStorage.length).toBe(0)
  await removePreparedSubmission('key', 'a')
  expect(native.update).toHaveBeenLastCalledWith('key', null)
  vi.stubGlobal('hermesDesktop', undefined)
  await writePreparedSubmission('key', entry)
  expect(await readPreparedSubmission('key')).toEqual(entry)
  await removePreparedSubmission('key', 'a')
  expect(await readPreparedSubmission('key')).toBeUndefined()
})
