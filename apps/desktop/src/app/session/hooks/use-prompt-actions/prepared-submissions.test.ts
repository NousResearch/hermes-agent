import { waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { claimPreparedSubmission, listPreparedDrafts, type PreparedSubmission, preparedSubmissionKey, readPreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
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

  const before = localStorage.getItem('hermes.desktop.preparedSubmissions.v1')
  expect((await listPreparedDrafts('stored', destination.scopeKey)).map(entry => entry.key)).toEqual([(await readPreparedSubmission(ordinary))!.journal!.storageKey])
  expect(await listPreparedDrafts('another', destination.scopeKey)).toEqual([])
  expect(localStorage.getItem('hermes.desktop.preparedSubmissions.v1')).toBe(before)
  expect((await readPreparedSubmission(slash))?.text).toBe('expanded skill instructions')
})

test('native preparation waits for acknowledgement and never downgrades a write failure to browser storage', async () => {
  let ack!: () => void
  const gate = new Promise<void>(resolve => { ack = resolve })
  const entry = { id: 'a', text: 'Ω\n  exact', attachments: [], params: { session_id: 'live' } } as unknown as PreparedSubmission
  const key = preparedSubmissionKey('stored', { scopeKey: 'local' } as SubmissionDestination, entry.text, [])
  const records: Record<string, unknown> = {}

  const native = { read: vi.fn(async () => JSON.stringify(records)), update: vi.fn(async (key: string, value: string | null) => {
    await gate

    if (value === null) {delete records[key]}
    else {records[key] = JSON.parse(value)}
  }) }

  vi.stubGlobal('hermesDesktop', { preparedSubmissions: native })
  let finished = false
  const writing = writePreparedSubmission(key, entry).then(() => { finished = true })
  await waitFor(() => expect(native.update).toHaveBeenCalledOnce())
  expect(finished).toBe(false)
  expect(native.update).toHaveBeenCalledWith(entry.journal!.storageKey, JSON.stringify(entry))
  ack(); await writing
  expect(await readPreparedSubmission(key)).toEqual(entry)
  native.update.mockRejectedValueOnce(new Error('disk full'))
  await expect(writePreparedSubmission(key, entry)).rejects.toThrow('disk full')
  expect(localStorage.length).toBe(0)
  await removePreparedSubmission(key, entry)
  expect(native.update).toHaveBeenLastCalledWith(entry.journal!.storageKey, null)
  vi.stubGlobal('hermesDesktop', undefined)
  const browserEntry = { ...entry, journal: undefined }
  await writePreparedSubmission(key, browserEntry)
  expect(await readPreparedSubmission(key)).toEqual(browserEntry)
  await removePreparedSubmission(key, browserEntry)
  expect(await readPreparedSubmission(key)).toBeUndefined()
})

test('explicit recovery rejects metadata changed since the displayed draft was listed', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const key = preparedSubmissionKey('stored', destination, 'text-only draft', [])
  await writePreparedSubmission(key, { id: 'text-draft', owner: undefined, text: 'text-only draft', attachments: [], params: { session_id: 'original' } })
  const [displayed] = await listPreparedDrafts('stored', destination.scopeKey)
  expect(displayed.text).toBe('text-only draft')
  const changed = (await readPreparedSubmission(key))!
  changed.params = { session_id: 'different' }
  await writePreparedSubmission(key, changed)
  await expect(claimPreparedSubmission(displayed.key, displayed.expected)).rejects.toThrow('changed during recovery')
})

test('browser writes fail before dispatch when atomic storage is unavailable', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  vi.stubGlobal('navigator', { locks: undefined })
  const key = preparedSubmissionKey('stored', { scopeKey: 'local' } as SubmissionDestination, 'text', [])
  await expect(writePreparedSubmission(key, { id: 'unsupported', owner: undefined, text: 'text', attachments: [], params: {} })).rejects.toThrow('Atomic draft storage unavailable')
  expect(localStorage.length).toBe(0)
})

test('long frozen input does not become a native journal address or get truncated', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const text = 'exact input\n'.repeat(2048)
  const key = preparedSubmissionKey('stored', { scopeKey: 'local' } as SubmissionDestination, text, [])
  const entry: PreparedSubmission = { id: 'short-uuid', owner: undefined, text, attachments: [], params: { text } }
  await writePreparedSubmission(key, entry)
  expect(entry.journal!.storageKey.length).toBeLessThan(100)
  expect((await readPreparedSubmission(key))?.params.text).toBe(text)
})
