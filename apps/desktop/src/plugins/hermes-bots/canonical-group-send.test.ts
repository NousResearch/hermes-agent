// @vitest-environment node
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
// Keep native implementation out of the renderer TypeScript project while
// exercising the real atomic-file implementation in this Node fixture.
const nativeModule = '../../../electron/prepared-submissions'
const { preparedJournal } = await import(/* @vite-ignore */ nativeModule)
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'

const binding = { connectionId: 'remote-a', profile: 'profile-a', roomId: 'room-a' }
afterEach(() => vi.unstubAllGlobals())

test('uncertain sends reopen with exact identity and attachments, isolated by authority, until matching ACK retirement', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'group-send-journal-'))
  const reopen = () => {
    const journal = preparedJournal(dir, 'http://localhost:5174')
    vi.stubGlobal('window', { hermesDesktop: { preparedSubmissions: {
      read: async () => JSON.stringify(journal.read()),
      update: async (key: string, entry: string | null) => journal.update(key, entry === null ? null : JSON.parse(entry))
    } } })
  }
  try {
    reopen()
    const payload = { text: 'Ω\n  exact', attachments: [{ path: '/owner/cache/image.png', mime_type: 'image/png' }] }
    const prepared = await prepareCanonicalGroupSend(binding, payload)
    expect(prepared.params.event_id).toBeTruthy()
    expect(prepared.params.payload.thread_id).toBe(prepared.params.event_id)
    const expected = JSON.parse(JSON.stringify(prepared))
    payload.attachments[0].path = '/mutated'
    reopen() // The server ACK was lost; no retirement has happened.
    expect(await readCanonicalGroupSend(binding)).toEqual(expected)
    expect(await prepareCanonicalGroupSend(binding, { text: 'new draft must not replace uncertain input' })).toEqual(expected)
    for (const other of [{ ...binding, profile: 'other' }, { ...binding, roomId: 'other' }, { ...binding, connectionId: 'other' }]) {
      expect(await readCanonicalGroupSend(other)).toBeUndefined()
    }
    await retireCanonicalGroupSend(binding, 'stale-ack')
    expect(await readCanonicalGroupSend(binding)).toEqual(expected)
    await retireCanonicalGroupSend(binding, expected.params.event_id)
    reopen()
    expect(await readCanonicalGroupSend(binding)).toBeUndefined()
    expect((await prepareCanonicalGroupSend(binding, { text: 'fresh intent' })).params.event_id).not.toBe(expected.params.event_id)
  } finally { fs.rmSync(dir, { recursive: true, force: true }) }
})

test('native journal acknowledgement gates send and a failed write cannot downgrade to browser storage', async () => {
  let acknowledge!: () => void
  let entered!: () => void
  const writing = new Promise<void>(resolve => { entered = resolve })
  const gate = new Promise<void>(resolve => { acknowledge = resolve })
  const native = { read: async () => '{}', update: vi.fn(() => { entered(); return gate }) }
  const browserWrite = vi.fn()
  vi.stubGlobal('window', { hermesDesktop: { preparedSubmissions: native }, localStorage: { setItem: browserWrite } })
  const send = vi.fn()
  const pending = prepareCanonicalGroupSend(binding, { text: 'before network' }).then(send)
  await writing
  expect(send).not.toHaveBeenCalled()
  acknowledge()
  await pending
  expect(send).toHaveBeenCalledOnce()
  native.update.mockRejectedValueOnce(new Error('disk full'))
  await expect(prepareCanonicalGroupSend(binding, { text: 'blocked' }).then(send)).rejects.toThrow('disk full')
  expect(send).toHaveBeenCalledOnce()
  expect(browserWrite).not.toHaveBeenCalled()
})
