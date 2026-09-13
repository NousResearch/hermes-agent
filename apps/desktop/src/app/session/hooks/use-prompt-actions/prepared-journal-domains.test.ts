import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { compareJournal, journalOwner } from './prepared-journal-storage'
import { claimPreparedSubmission, type PreparedSubmission, preparedSubmissionKey, readPreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
import type { SubmissionDestination } from './submission-destination'

const previousNative = window.hermesDesktop
let home: string | undefined

afterEach(() => {
  vi.unstubAllGlobals()
  window.hermesDesktop = previousNative
  localStorage.clear()

  if (home) {fs.rmSync(home, { recursive: true, force: true }); home = undefined}
  vi.doUnmock('electron')
})

async function journal(kind: 'native' | 'browser') {
  if (kind === 'browser') {
    vi.stubGlobal('hermesDesktop', undefined)

    return {
      read: () => JSON.parse(localStorage.getItem('hermes.desktop.preparedSubmissions.v1') || '{}'),
      seed: (key: string, entry: unknown) => localStorage.setItem('hermes.desktop.preparedSubmissions.v1', JSON.stringify({ [key]: entry }))
    }
  }

  vi.doMock('electron', () => ({ app: {}, ipcMain: {} }))
  const nativeModule = '../../../../../electron/prepared-submissions'
  const { preparedJournal } = await import(nativeModule)
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-domains-'))
  const store = preparedJournal(home, 'http://same-origin')

  const compare = async (key: string, expected: string | null, entry: string | null) =>
    store.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))

  window.hermesDesktop = { ...previousNative, preparedSubmissions: {
    owner: async () => 'same-window', read: async () => JSON.stringify(store.read()),
    update: async (key, entry) => store.update(key, entry === null ? null : JSON.parse(entry)),
    compareAndSet: compare, compareSend: compare
  } } as typeof window.hermesDesktop

  return { read: store.read, seed: store.update }
}

function input(scope: string, session: string, text = 'same text') {
  const destination = { scopeKey: scope, owner: { connectionId: scope, profile: 'named' } } as SubmissionDestination
  const key = preparedSubmissionKey(session, destination, text, [], { submission_id: 'caller-owned-id' })

  const entry: PreparedSubmission = {
    id: 'caller-owned-id', owner: destination.owner, text, attachments: [],
    params: { session_id: session, submission_id: 'caller-owned-id', text }
  }

  return { key, entry }
}

it.each(['native', 'browser'] as const)('%s: caller IDs are independent across target and owner domains', async kind => {
  const store = await journal(kind)
  const inputs = [input('owner-a', 'session-a'), input('owner-a', 'session-b'), input('owner-b', 'session-a')]

  for (const { key, entry } of inputs) {await writePreparedSubmission(key, entry)}
  expect(new Set(inputs.map(({ entry }) => entry.journal!.storageKey)).size).toBe(3)
  expect(Object.keys(store.read())).toHaveLength(3)

  for (const { key, entry } of inputs) {
    const restored = (await readPreparedSubmission(key))!
    expect(restored.id).toBe('caller-owned-id')
    expect(restored.owner).toEqual(entry.owner)
    expect(restored.params).toEqual(entry.params)
  }

  await removePreparedSubmission(inputs[0].key, inputs[0].entry)
  expect(Object.keys(store.read())).toHaveLength(2)
  expect((await readPreparedSubmission(inputs[1].key))?.params).toEqual(inputs[1].entry.params)
  expect((await readPreparedSubmission(inputs[2].key))?.params).toEqual(inputs[2].entry.params)
})

it.each(['native', 'browser'] as const)('%s: same-domain reuse cannot create a different intent by changing content', async kind => {
  const store = await journal(kind)
  const first = input('owner-a', 'session-a')
  const second = input('owner-a', 'session-a', 'different text')
  await writePreparedSubmission(first.key, first.entry)
  await expect(writePreparedSubmission(second.key, second.entry)).rejects.toThrow('changed before write')
  expect(Object.keys(store.read())).toHaveLength(1)
  expect((await readPreparedSubmission(first.key))?.params).toEqual(first.entry.params)
})

it.each(['native', 'browser'] as const)('%s: existing intent-only records recover in place alongside a new domain', async kind => {
  const store = await journal(kind)
  const original = input('owner-a', 'session-a')
  const oldKey = JSON.stringify(['intent', original.entry.id])
  const originalOwner = await journalOwner()
  original.entry.journal = { lookup: original.key, storageKey: oldKey, owner: originalOwner }
  store.seed(oldKey, original.entry)
  const other = input('owner-b', 'session-b')
  await writePreparedSubmission(other.key, other.entry)
  const retained = (await readPreparedSubmission(original.key))!
  expect(retained.journal!.storageKey).toBe(oldKey)
  await writePreparedSubmission(original.key, retained)
  await claimPreparedSubmission(oldKey, JSON.stringify(retained))
  const recovered = (await readPreparedSubmission(original.key))!
  expect(recovered.id).toBe(original.entry.id)
  expect(recovered.params).toEqual(original.entry.params)
  expect(recovered.journal!.storageKey).toBe(oldKey)
  await removePreparedSubmission(original.key, recovered)
  expect(store.read()[oldKey]).toBeUndefined()
  expect((await readPreparedSubmission(other.key))?.params).toEqual(other.entry.params)
})

it.each(['native', 'browser'] as const)('%s: large legacy content-key records keep their original ID and address', async kind => {
  const store = await journal(kind)
  const caption = 'retained caption '.repeat(256)
  const original = input('owner-a', 'session-a', caption)
  original.entry.params.preview = 'x'.repeat(1024 * 1024)
  store.seed(original.key, original.entry)
  await claimPreparedSubmission(original.key, JSON.stringify(original.entry))
  const recovered = (await readPreparedSubmission(original.key))!
  expect(recovered.id).toBe(original.entry.id)
  expect(recovered.journal!.storageKey).toBe(original.key)
  expect(recovered.params).toEqual(original.entry.params)
  await writePreparedSubmission(original.key, recovered)
  expect(Object.keys(store.read())).toEqual([original.key])
  await removePreparedSubmission(original.key, recovered)
  expect(store.read()).toEqual({})
})

it('never retries ordinary CAS refusal through Group CAS, unconditional update or browser storage', async () => {
  const compareSend = vi.fn(async () => false)
  const compareAndSet = vi.fn()
  const update = vi.fn()
  window.hermesDesktop = { ...previousNative, preparedSubmissions: {
    read: async () => '{}', compareSend, compareAndSet, update
  } }
  expect(await compareJournal('ordinary', null, '{}', false, true)).toBe(false)
  compareSend.mockRejectedValueOnce(new Error('fixture CAS storage failure'))
  await expect(compareJournal('ordinary', null, '{}', false, true)).rejects.toThrow('fixture CAS storage failure')
  expect(compareAndSet).not.toHaveBeenCalled()
  expect(update).not.toHaveBeenCalled()
  expect(localStorage.length).toBe(0)
})

it('missing ordinary CAS cannot be replaced by bounded Group CAS for caller IDs or recovery', async () => {
  const compareAndSet = vi.fn()
  const update = vi.fn()
  window.hermesDesktop = { ...previousNative, preparedSubmissions: { read: async () => '{}', compareAndSet, update } }
  const { key, entry } = input('owner', 'session')
  await expect(writePreparedSubmission(key, entry)).rejects.toThrow('Atomic draft storage unavailable')
  await expect(compareJournal('existing', '{}', '{}', true)).rejects.toThrow('Atomic draft storage unavailable')
  await expect(compareJournal('existing', '{}', null)).rejects.toThrow('Atomic draft storage unavailable')
  expect(compareAndSet).not.toHaveBeenCalled()
  expect(update).not.toHaveBeenCalled()
})

it('an update-only bridge permits only first publication of a fresh renderer-generated intent', async () => {
  const records: Record<string, unknown> = {}
  const update = vi.fn(async (key: string, entry: string | null) => {records[key] = JSON.parse(entry!)})
  window.hermesDesktop = { ...previousNative, preparedSubmissions: { read: async () => JSON.stringify(records), update } }
  const key = JSON.stringify(['intent-v2', 'scope', 'session', crypto.randomUUID()])
  expect(await compareJournal(key, null, '{}', false, true)).toBe(true)
  expect(await compareJournal(key, null, '{}', false, true)).toBe(false)
  await expect(compareJournal(key, '{}', '{}', false, true)).rejects.toThrow('Atomic draft storage unavailable')
  await expect(compareJournal(key, '{}', null, false, true)).rejects.toThrow('Atomic draft storage unavailable')
  expect(update).toHaveBeenCalledOnce()
})
