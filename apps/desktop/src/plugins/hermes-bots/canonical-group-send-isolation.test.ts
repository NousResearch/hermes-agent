// @vitest-environment node
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
const nativeModule = '../../../electron/prepared-submissions'
const { preparedJournal } = await import(/* @vite-ignore */ nativeModule)
import { attemptCanonicalGroupSend, claimCanonicalGroupSend, listCanonicalGroupSends, prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend, settleCanonicalGroupSend } from './canonical-group-send'

const binding = { connectionId: 'local', profile: 'default', roomId: 'window-room' }
const directories: string[] = []
afterEach(() => { vi.unstubAllGlobals();

 for (const directory of directories.splice(0)) {fs.rmSync(directory, { recursive: true, force: true })} })

function fixture() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'canonical-send-isolation-'))
  directories.push(directory)
  const store = preparedJournal(directory, 'http://localhost:5174')

  const bind = (owner: string) => vi.stubGlobal('window', { hermesDesktop: { preparedSubmissions: {
    owner: async () => owner,
    read: async () => JSON.stringify(store.read()),
    update: async (key: string, value: string | null) => store.update(key, value === null ? null : JSON.parse(value)),
    compareSend: async (key: string, expected: string | null, value: string | null) =>
      store.compareAndSet(key, expected === null ? null : JSON.parse(expected), value === null ? null : JSON.parse(value))
  } } })

  return { store, bind }
}

it('a different window prepares its own message without adopting another unknown intent', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const a = await prepareCanonicalGroupSend(binding, { text: 'First window', attachments: [] })
  bind('window-b')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
  const b = await prepareCanonicalGroupSend(binding, { text: 'Second window', attachments: [] })

  expect(b.params.payload.text).toBe('Second window')
  expect(b.params.event_id).not.toBe(a.params.event_id)
  expect(Object.values(store.read())).toHaveLength(2)
  bind('window-a')
  expect(await readCanonicalGroupSend(binding)).toEqual(a)
})

it('concurrent window preparations retain both exact identities instead of overwriting one slot', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const first = prepareCanonicalGroupSend(binding, { text: 'A' })
  bind('window-b')
  const second = prepareCanonicalGroupSend(binding, { text: 'B' })
  const [a, b] = await Promise.all([first, second])

  expect(a.params.event_id).not.toBe(b.params.event_id)
  expect(Object.values(store.read()).map((entry: any) => entry.params.event_id).sort()).toEqual([a.params.event_id, b.params.event_id].sort())
})

it('another window cannot retire a captured intent after a delayed acknowledgement', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const a = await prepareCanonicalGroupSend(binding, { text: 'Retain this' })
  bind('window-b')
  await expect(retireCanonicalGroupSend(binding, a.params.event_id, a)).rejects.toThrow(/ownership/i)
  expect(Object.values(store.read())).toEqual([a])
})

it('a changed receipt snapshot cannot be deleted merely because its event ID still matches', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const a = await prepareCanonicalGroupSend(binding, { text: 'Original' })
  const storageKey = Object.keys(store.read())[0]
  const changed = { ...a, params: { ...a.params, payload: { text: 'Changed while pending' } } }
  store.update(storageKey, changed)

  await expect(retireCanonicalGroupSend(binding, a.params.event_id, a)).rejects.toThrow(/changed/i)
  expect(store.read()[storageKey]).toEqual(changed)
})

it('a cold window offers explicit recovery, transfers the original identity, and fences the former window', async () => {
  const { store, bind } = fixture()
  bind('closed-window')
  const a = await prepareCanonicalGroupSend(binding, { text: 'Ω\n original', attachments: [{ name: 'exact.txt' }] })
  bind('new-window')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
  const [offered] = await listCanonicalGroupSends(binding)
  expect(offered.entry.params).toEqual(a.params)
  expect(Object.values(store.read())[0]).toEqual(a)
  const claimed = await claimCanonicalGroupSend(binding, offered)
  expect(claimed.params).toEqual(a.params)
  expect(await readCanonicalGroupSend(binding)).toEqual(claimed)
  bind('closed-window')
  await expect(attemptCanonicalGroupSend(binding, a)).rejects.toThrow('changed before Retry')
  await expect(retireCanonicalGroupSend(binding, a.params.event_id, a)).rejects.toThrow('changed')
  expect(Object.values(store.read())[0]).toEqual(claimed)
})

it('preserves legacy room-slot input until an explicit exact recovery', async () => {
  const { store, bind } = fixture()
  const legacy = { binding, params: { room_id: binding.roomId, event_id: 'legacy-frozen-id', payload: { text: 'Legacy pending text', thread_id: 'legacy-frozen-id' } } }
  const key = JSON.stringify(['canonical-group-send-v1', binding.connectionId, binding.profile, binding.roomId])
  store.update(key, legacy)
  bind('new-window')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
  const [offered] = await listCanonicalGroupSends(binding)
  expect(offered.entry).toEqual(legacy)
  const claimed = await claimCanonicalGroupSend(binding, offered)
  expect(claimed.params).toEqual(legacy.params)
  expect(claimed.journal!.storageKey).toBe(key)
  expect(await attemptCanonicalGroupSend(binding, claimed)).toBe(false)
  expect((store.read()[key] as { attempted: boolean }).attempted).toBe(true)
})

it('does not claim a changed recovery selection or another destination', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const a = await prepareCanonicalGroupSend(binding, { text: 'A' })
  bind('window-b')
  const [offered] = await listCanonicalGroupSends(binding)
  expect(await listCanonicalGroupSends({ ...binding, profile: 'other' })).toEqual([])
  store.update(offered.storageKey, { ...a, params: { ...a.params, payload: { text: 'changed' } } })
  await expect(claimCanonicalGroupSend(binding, offered)).rejects.toThrow('changed')
})

it('local acknowledgement failure retains the exact receipt without making another window cleanup authority', async () => {
  const { store, bind } = fixture()
  bind('window-a')
  const a = await prepareCanonicalGroupSend(binding, { text: 'Accepted A' })
  bind('window-b')
  const [offered] = await listCanonicalGroupSends(binding)
  const b = await claimCanonicalGroupSend(binding, offered)
  bind('window-a')
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

  try {
    await settleCanonicalGroupSend(binding, a)
    expect(Object.values(store.read())).toEqual([b])
    expect(await readCanonicalGroupSend(binding)).toBeUndefined()
    const next = await prepareCanonicalGroupSend(binding, { text: 'New A' })
    expect(next.params.event_id).not.toBe(a.params.event_id)
    expect(Object.values(store.read())).toHaveLength(2)
  } finally {warn.mockRestore()}
})

it('refuses a native bridge missing owner/CAS rather than using its unsafe update or browser storage', async () => {
  const update = vi.fn(), browser = vi.fn()
  vi.stubGlobal('window', { hermesDesktop: { preparedSubmissions: { read: async () => '{}', update } }, localStorage: { setItem: browser } })
  await expect(prepareCanonicalGroupSend(binding, { text: 'must stay unsent' })).rejects.toThrow('update Desktop')
  expect(update).not.toHaveBeenCalled()
  expect(browser).not.toHaveBeenCalled()
})

it.each(['entire journal', 'read', 'owner', 'compareSend'])('refuses incomplete native storage: missing %s', async missing => {
  const update = vi.fn(), browserRead = vi.fn(() => '{}'), browserWrite = vi.fn()
  const native: Record<string, unknown> = { read: async () => '{}', owner: async () => 'window', compareSend: vi.fn(), update }

  delete native[missing]
  vi.stubGlobal('window', { hermesDesktop: missing === 'entire journal' ? {} : { preparedSubmissions: native },
    localStorage: { getItem: browserRead, setItem: browserWrite } })
  vi.stubGlobal('navigator', { locks: { request: vi.fn((_name, run) => run()) } })
  await expect(prepareCanonicalGroupSend(binding, { text: 'must stay unsent' })).rejects.toThrow('update Desktop')
  expect(update).not.toHaveBeenCalled()
  expect(browserRead).not.toHaveBeenCalled()
  expect(browserWrite).not.toHaveBeenCalled()
})

it('publishes uncertainty before dispatch and never restores an attempted record as fresh', async () => {
  const { store, bind } = fixture()
  bind('window')
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Durable before effect' })
  const native = window.hermesDesktop!.preparedSubmissions!
  const compare = native.compareSend!
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  native.compareSend = vi.fn(async (...args: Parameters<typeof compare>) => { await gate; return compare(...args) })
  const dispatch = vi.fn(() => expect((Object.values(store.read())[0] as { attempted: boolean }).attempted).toBe(true))
  const attempting = attemptCanonicalGroupSend(binding, entry).then(fresh => { dispatch(); return fresh })
  await Promise.resolve()
  expect(dispatch).not.toHaveBeenCalled()
  expect((Object.values(store.read())[0] as { attempted: boolean }).attempted).toBe(false)
  release()
  expect(await attempting).toBe(true)
  bind('window')
  const reopened = (await readCanonicalGroupSend(binding))!
  expect(reopened.params).toEqual(entry.params)
  expect(await attemptCanonicalGroupSend(binding, reopened)).toBe(false)
})

it('keeps an intent fresh and unsent when publishing uncertainty fails', async () => {
  const { store, bind } = fixture()
  bind('window')
  const entry = await prepareCanonicalGroupSend(binding, { text: 'No durable attempt' })
  window.hermesDesktop!.preparedSubmissions!.compareSend = vi.fn().mockRejectedValue(new Error('disk full'))
  const dispatch = vi.fn()
  await expect(attemptCanonicalGroupSend(binding, entry).then(dispatch)).rejects.toThrow('disk full')
  expect(dispatch).not.toHaveBeenCalled()
  expect(entry.attempted).toBe(false)
  expect(Object.values(store.read())).toEqual([entry])
})
