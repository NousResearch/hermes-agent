import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type * as HermesSdk from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
const nativeModule = '../../../electron/prepared-submissions'
const { preparedJournal } = await import(/* @vite-ignore */ nativeModule)
vi.mock('@hermes/plugin-sdk', async () => {
  const sdk = await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk')
  const { pluginSdkMock, createGroupGateway, captureGroupRequests } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { ...sdk, ...await pluginSdkMock(gateway.host), atom, useValue: useStore, MessageTextContent: sdk.MessageTextContent,
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request } }
})
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings, $canonicalGroupNames, registerCanonicalGroup } from './canonical-group-registry'
import { claimCanonicalGroupSend, listCanonicalGroupSends, prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { GroupChatWorkspace } from './group-chat-view'
import { CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'
const originalDesktop = window.hermesDesktop
const labels = CANONICAL_GROUP_LOCALES.en

const chooseGroupAction = async (name: string) => {
  fireEvent.pointerDown(await screen.findByRole('button', { name: labels.groupActions }), { button: 0, ctrlKey: false })
  fireEvent.click(await screen.findByRole('menuitem', { name }))
}

beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop })

it.each(['entire journal', 'read', 'owner', 'compareSend'])('does not Send or write browser storage after incomplete native storage: missing %s', async missing => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: `missing-native-${missing}` }
  const compareSend = vi.fn(), update = vi.fn()
  const native: Record<string, unknown> = { read: async () => '{}', owner: async () => 'native-window', compareSend, update }
  const desktop: Record<string, unknown> = { preparedSubmissions: native }
  window.hermesDesktop = desktop as unknown as typeof window.hermesDesktop
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : { accepted: true, client_event_id: params.event_id })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))

  if (missing === 'entire journal') {delete desktop.preparedSubmissions}
  else {delete native[missing]}
  const browserRead = vi.spyOn(Storage.prototype, 'getItem'), browserWrite = vi.spyOn(Storage.prototype, 'setItem')
  try {
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep this unsent' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText('Atomic draft storage unavailable; update Desktop', {}, { timeout: 1000 })
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
    expect(browserRead).not.toHaveBeenCalled()
    expect(browserWrite).not.toHaveBeenCalled()
    expect(update).not.toHaveBeenCalled()
    expect(compareSend).not.toHaveBeenCalled()
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Keep this unsent')
  } finally {browserRead.mockRestore(); browserWrite.mockRestore()}
})

it.each(['mounted', 'remount', 'transfer', 'legacy'].flatMap(mode =>
  ['permission_denied', 'stale_generation'].map(reason => ({ mode, reason }))))('$mode keeps the unknown Send identity after a later $reason refusal', async ({ mode, reason }) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'group-monotonic-ui-'))
  const journal = preparedJournal(directory, 'http://localhost:5174')
  const bind = (owner: string) => { window.hermesDesktop = { preparedSubmissions: {
    owner: async () => owner, read: async () => JSON.stringify(journal.read()), update: vi.fn(),
    compareSend: async (key: string, expected: string | null, entry: string | null) =>
      journal.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  } } as unknown as typeof window.hermesDesktop }
  const binding = { connectionId: 'remote', profile: 'team', roomId: `monotonic-${mode}-${reason}` }
  const attempted: string[] = [], markers: unknown[] = []
  const accepted = new Map<string, unknown>()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}
    if (method === 'groups.log') {return { events: [] }}
    attempted.push(params.event_id)
    markers.push((Object.values(journal.read())[0] as { attempted?: boolean })?.attempted)
    // Controlled peer acceptance, then transport loss; no gateway/model process.
    if (mode !== 'legacy' && attempted.length === 1) {
      accepted.set(params.event_id, params.payload)
      throw new Error('acceptance reply lost')
    }
    throw Object.assign(new Error(reason), { code: 4001, data: { reason } })
  })
  try {
    bind('original-window')
    let originalId: string
    if (mode === 'legacy') {
      originalId = 'legacy-unknown-id'
      const entry = { binding, params: { room_id: binding.roomId, event_id: originalId, payload: { text: 'Original accepted intent', attachments: [], thread_id: originalId } } }
      journal.update(JSON.stringify(['canonical-group-send-v1', binding.connectionId, binding.profile, binding.roomId]), entry)
      accepted.set(originalId, entry.params.payload)
    } else {
      const first = render(<CanonicalGroupWorkspace binding={binding} />)
      await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Original accepted intent' } })
      fireEvent.click(screen.getByRole('button', { name: 'Send' }))
      await screen.findByText('acceptance reply lost')
      originalId = attempted[0]
      if (mode !== 'mounted') {first.unmount()}
    }

    if (mode !== 'mounted') {
      if (mode === 'transfer') {bind('new-window')}
      render(<CanonicalGroupWorkspace binding={binding} />)
      if (mode === 'transfer' || mode === 'legacy') {
        fireEvent.click(await screen.findByRole('button', { name: 'Restore draft' }))
      }
    }
    const retry = await screen.findByRole('button', { name: 'Retry' })
    await waitFor(() => expect((retry as HTMLButtonElement).disabled).toBe(false))
    const before = attempted.length
    fireEvent.click(retry)
    await screen.findByText(reason)
    expect(Object.values(journal.read()).map((entry: any) => ({ id: entry.params.event_id, text: entry.params.payload.text })))
      .toEqual([{ id: originalId, text: 'Original accepted intent' }])
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Original accepted intent')
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true)
    expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.sendMaybe)).toBeTruthy()
    const again = screen.getByRole('button', { name: 'Retry' })
    await waitFor(() => expect((again as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(again)
    await waitFor(() => expect(attempted).toHaveLength(before + 2))
    expect(attempted.every(id => id === originalId)).toBe(true)
    expect(markers.every(marker => marker === true)).toBe(true)
    expect([...accepted.keys()]).toEqual([originalId])
  } finally {cleanup(); fs.rmSync(directory, { recursive: true, force: true })}
})

it.each(['permission_denied', 'stale_generation'])('keeps a fresh proven %s refusal editable', async reason => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: `fresh-${reason}` }
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : Promise.reject(Object.assign(new Error(reason), { code: 4001, data: { reason } })))
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Fresh refused intent' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.sendRefused)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Fresh refused intent')
})

it.each([null, 0, 1, 'true', [], {}])('keeps the exact prepared message on malformed provided acceptance %j', async accepted => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'invalid-acceptance' }
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : { accepted, client_event_id: params.event_id })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep this exact message' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))

  await screen.findByRole('button', { name: 'Retry' })
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Keep this exact message'))
  const retained = await readCanonicalGroupSend(binding)
  expect(retained?.params.payload.text).toBe('Keep this exact message')
  expect(request.mock.calls.filter(call => call[1] === 'groups.send')).toHaveLength(1)
  expect(request.mock.calls.find(call => call[1] === 'groups.send')![2].event_id).toBe(retained!.params.event_id)
})

it('retains legacy acknowledgement compatibility when accepted is absent', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'legacy-acceptance' }
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : { client_event_id: params.event_id })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Compatible receipt' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
})

it('a second already-mounted window sends its own text while the first acknowledgement is unknown', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'group-window-ui-'))
  const journal = preparedJournal(directory, 'http://localhost:5174')
  const bind = (owner: string) => { window.hermesDesktop = { preparedSubmissions: {
    owner: async () => owner, read: async () => JSON.stringify(journal.read()), update: vi.fn(),
    compareSend: async (key: string, expected: string | null, entry: string | null) =>
      journal.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  } } as unknown as typeof window.hermesDesktop }
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'two-window' }
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}
    if (method === 'groups.log') {return { events: [] }}
    if (params.payload.text === 'Window A') {throw new Error('lost ACK')}
    return { accepted: true, client_event_id: params.event_id }
  })
  try {
    bind('window-a')
    const a = render(<CanonicalGroupWorkspace binding={binding} />)
    await waitFor(() => expect((within(a.container).getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
    bind('window-b')
    const b = render(<CanonicalGroupWorkspace binding={binding} />)
    await waitFor(() => expect((within(b.container).getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
    bind('window-a')
    fireEvent.change(within(a.container).getByRole('textbox'), { target: { value: 'Window A' } })
    fireEvent.click(within(a.container).getByRole('button', { name: 'Send' }))
    await within(a.container).findByText('lost ACK')
    const first = request.mock.calls.find(call => call[1] === 'groups.send')![2]
    bind('window-b')
    fireEvent.change(within(b.container).getByRole('textbox'), { target: { value: 'Window B' } })
    fireEvent.click(within(b.container).getByRole('button', { name: 'Send' }))
    await waitFor(() => expect((within(b.container).getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
    const sends = request.mock.calls.filter(call => call[1] === 'groups.send')
    expect(sends.map(call => call[2].payload.text)).toEqual(['Window A', 'Window B'])
    expect(sends[1][2].event_id).not.toBe(first.event_id)
    expect(Object.values(journal.read()).map((entry: any) => entry.params.event_id)).toEqual([first.event_id])
  } finally {fs.rmSync(directory, { recursive: true, force: true })}
})

it('offers a cold-window restore without sending, preserves a newer draft, then retries the chosen original identity', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'group-recovery-ui-'))
  const journal = preparedJournal(directory, 'http://localhost:5174')
  const bind = (owner: string) => { window.hermesDesktop = { preparedSubmissions: {
    owner: async () => owner, read: async () => JSON.stringify(journal.read()), update: vi.fn(),
    compareSend: async (key: string, expected: string | null, entry: string | null) =>
      journal.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  } } as unknown as typeof window.hermesDesktop }
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'chosen-recovery' }
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : { accepted: true, client_event_id: params.event_id })
  try {
    bind('closed-window')
    const original = await prepareCanonicalGroupSend(binding, { text: 'Frozen original', attachments: [] })
    bind('new-window')
    render(<CanonicalGroupWorkspace binding={binding} />)
    await screen.findByRole('button', { name: 'Restore draft' })
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('')
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Newer draft' } })
    expect((screen.getByRole('button', { name: 'Restore draft' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Newer draft')
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Restore draft' }))
    await screen.findByRole('button', { name: 'Retry' })
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
    const sent = request.mock.calls.find(call => call[1] === 'groups.send')![2]
    expect(sent).toEqual({ ...original.params, profile: binding.profile })
  } finally {fs.rmSync(directory, { recursive: true, force: true })}
})

it('keeps Restore unclaimed during upload and while the new attachment occupies the composer', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'group-upload-recovery-'))
  const journal = preparedJournal(directory, 'http://localhost:5174')
  const compare = vi.spyOn(journal, 'compareAndSet')
  const bind = (owner: string) => { window.hermesDesktop = { preparedSubmissions: {
    owner: async () => owner, read: async () => JSON.stringify(journal.read()), update: vi.fn(),
    compareSend: async (key: string, expected: string | null, entry: string | null) =>
      journal.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  } } as unknown as typeof window.hermesDesktop }
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'upload-recovery' }
  let releaseUpload!: (value: unknown) => void
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}
    if (method === 'groups.log') {return { events: [] }}
    if (method === 'groups.attachment.upload') {return new Promise(resolve => { releaseUpload = resolve })}
    return {}
  })
  try {
    bind('original-window')
    const original = await prepareCanonicalGroupSend(binding, { text: 'Frozen original', attachments: [] })
    compare.mockClear()
    bind('new-window')
    const view = render(<CanonicalGroupWorkspace binding={binding} />)
    const restore = await screen.findByRole('button', { name: labels.restorePendingSend }) as HTMLButtonElement
    fireEvent.change(view.container.querySelector('input[type=file]')!, { target: { files: [new File(['A'], 'notes.txt', { type: 'text/plain' })] } })
    await waitFor(() => expect(releaseUpload).toBeTypeOf('function'))
    expect(restore.disabled).toBe(true)
    fireEvent.click(restore)
    expect(compare).not.toHaveBeenCalled()
    expect(Object.values(journal.read())).toEqual([original])
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
    await act(async () => releaseUpload({ attachment_id: 'uploaded', kind: 'file', name: 'notes.txt', mime: 'text/plain' }))
    expect(restore.disabled).toBe(true)
    fireEvent.click(restore)
    expect(compare).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: labels.removeAttachment }))
    await waitFor(() => expect(restore.disabled).toBe(false))
    fireEvent.click(restore)
    await screen.findByRole('button', { name: 'Retry' })
    expect((Object.values(journal.read())[0] as { params: unknown }).params).toEqual(original.params)
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
  } finally {cleanup(); fs.rmSync(directory, { recursive: true, force: true })}
})

it('a queued editor update cannot be cleared by a prior message acknowledgement', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'newer-edit' }
  let acknowledge!: () => void
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}
    if (method === 'groups.log') {return { events: [] }}
    await new Promise<void>(resolve => {acknowledge = resolve})
    return { accepted: true, client_event_id: params.event_id }
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Sent text' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(acknowledge).toBeTypeOf('function'))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Queued newer edit' } })
  acknowledge()
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.state').length).toBeGreaterThan(1))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Queued newer edit')
  expect(request.mock.calls.find(call => call[1] === 'groups.send')![2].payload.text).toBe('Sent text')
})

it('a mounted former window cannot Retry after another window explicitly claims its intent', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'group-transferred-ui-'))
  const journal = preparedJournal(directory, 'http://localhost:5174')
  const bind = (owner: string) => { window.hermesDesktop = { preparedSubmissions: {
    owner: async () => owner, read: async () => JSON.stringify(journal.read()), update: vi.fn(),
    compareSend: async (key: string, expected: string | null, entry: string | null) =>
      journal.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  } } as unknown as typeof window.hermesDesktop }
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'transferred' }
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] }
      : Promise.reject(new Error('lost ACK')))
  try {
    bind('window-a')
    const original = await prepareCanonicalGroupSend(binding, { text: 'Original intent', attachments: [] })
    render(<CanonicalGroupWorkspace binding={binding} />)
    await screen.findByRole('button', { name: 'Retry' })
    bind('window-b')
    const [offered] = await listCanonicalGroupSends(binding)
    const transferred = await claimCanonicalGroupSend(binding, offered)
    bind('window-a')
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('Prepared draft changed before Retry')
    expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
    expect(Object.values(journal.read())).toEqual([transferred])
    expect(transferred.params.event_id).toBe(original.params.event_id)
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Original intent')
  } finally {fs.rmSync(directory, { recursive: true, force: true })}
})

it('restores a frozen send after remount and retires only its acknowledged exact retry', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'restore' }
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Original', attachments: [{ path: '/owner/image.png', mime_type: 'image/png' }] })
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {throw new Error('lost ACK')}

    return {}
  })
  const first = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Original'))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true)
  expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('lost ACK')
  expect(await readCanonicalGroupSend(binding)).toEqual({ ...entry, attempted: true })
  first.unmount()
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByRole('button', { name: 'Retry' })
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : { accepted: true, client_event_id: params.event_id })
  await waitFor(() => expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  const sends = request.mock.calls.filter(c => c[1] === 'groups.send')
  expect(sends).toHaveLength(2)

  for (const call of sends) {
    expect(call[0]).toMatchObject({ connectionId: binding.connectionId, targetProfile: binding.profile })
    expect(call[2]).toEqual({ ...entry.params, profile: binding.profile })
  }
})

it('blocks Send until journal restore and durable preparation complete', async () => {
  let releaseRead!: (value: string) => void
  let releaseWrite!: () => void
  const journal: Record<string, unknown> = {}

  const native = {
    owner: async () => 'window',
    read: vi.fn().mockImplementationOnce(() => new Promise<string>(resolve => { releaseRead = resolve }))
      .mockImplementation(async () => JSON.stringify(journal)),
    update: vi.fn(),
    compareSend: vi.fn(async (key: string, expected: string | null, value: string | null) => {
      await new Promise<void>(resolve => { releaseWrite = resolve })

      if ((Object.hasOwn(journal, key) ? JSON.stringify(journal[key]) : null) !== expected) {return false}
      if (value === null) {delete journal[key]} else {journal[key] = JSON.parse(value)}

      return true
    })
  }

  window.hermesDesktop = { preparedSubmissions: native } as unknown as typeof window.hermesDesktop
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : { accepted: true, client_event_id: params.event_id })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'new' }} />)
  await screen.findByText('Room')
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'New text' } })
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true)
  releaseRead('{}')
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'New text' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(native.compareSend).toHaveBeenCalled())
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  releaseWrite()
  await waitFor(() => expect(native.compareSend).toHaveBeenCalledTimes(2))
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  expect((Object.values(journal)[0] as { attempted: boolean }).attempted).toBe(false)
  releaseWrite()
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  expect((Object.values(journal)[0] as { attempted: boolean }).attempted).toBe(true)
  await waitFor(() => expect(native.compareSend).toHaveBeenCalledTimes(3))
  releaseWrite()
  await waitFor(() => expect(native.compareSend).toHaveBeenCalledTimes(4))
  releaseWrite()
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
})

it('captures exact pending attempts through confirmation and never retargets or retries unknown work', async () => {
  const action = { kind: 'discard', member_id: 'worker', task_id: 'old-task', execution_generation: 7 }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { pending_actions: [action] } }}

    if (method === 'groups.log') {return { events: [], has_more: false }}

    if (method === 'groups.discard') {throw new Error('stale_attempt')}

    return {}
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'ack-room' }} />)
  fireEvent.click(await screen.findByRole('button', { name: labels.skipReply }))
  expect(screen.getByText(labels.discardWarning)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  action.execution_generation = 8
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDiscard }))
  await screen.findByText(labels.pendingActionUnconfirmed)
  expect(screen.getByRole('dialog')).toBeTruthy()
  const call = request.mock.calls.find(c => c[1] === 'groups.discard')!
  expect(call[0]).toMatchObject({ connectionId: 'remote', targetProfile: 'team' })
  expect(call[2]).toEqual({ room_id: 'ack-room', member_id: 'worker', task_id: 'old-task', execution_generation: 7, profile: 'team' })
  expect(request.mock.calls.every(c => c[1].startsWith('groups.'))).toBe(true)
})

it('reads back retry on the same authority and sends only through the group driver', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { pending_actions: [{ kind: 'retry', member_id: 'w', task_id: 't', execution_generation: 2 }] } }}

    if (method === 'groups.log') {return { events: [{ seq: 1, kind: 'message', payload: { text: 'Owner reply' } }], has_more: false }}

    return {}
  })
  const group = registerCanonicalGroup({ connectionId: 'local', profile: 'default' }, { room_id: 'r', name: 'Room', members: [] })
  render(<GroupChatWorkspace group={group} members={[]} />)
  fireEvent.click(await screen.findByRole('button', { name: labels.retryReply }))
  await waitFor(() => expect(request.mock.calls.filter(c => c[1] === 'groups.state').length).toBeGreaterThan(1))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  expect(screen.getByText('Owner reply')).toBeTruthy()
  expect(request.mock.calls.every(c => c[1].startsWith('groups.'))).toBe(true)
})

const settle = async () => {
  for (let turn = 0; turn < 10; turn++) {await act(async () => { await vi.advanceTimersByTimeAsync(0) })}
}

it('reads only new room log entries on each visible poll and restarts on a new authority epoch', async () => {
  vi.useFakeTimers()

  try {
    let epoch = 1
    let log = [{ seq: 1, kind: 'message', payload: { text: 'one' } }, { seq: 2, kind: 'message', payload: { text: 'two' } }]
    request.mockImplementation(async (_route, method, params) => {
      if (method === 'groups.state') {return { room: { name: 'Room', authority_epoch: epoch }, driver_status: {} }}

      if (method === 'groups.log') {return { events: log.filter(event => event.seq > params.since_seq), has_more: false }}

      return {}
    })
    const binding = { connectionId: 'remote', profile: 'team', roomId: 'poll-room' }
    const view = render(<CanonicalGroupWorkspace binding={binding} />)
    const reads = () => request.mock.calls.filter(call => call[1] === 'groups.log').map(call => call[2].since_seq)
    await settle()
    expect(reads()).toEqual([0])
    log = [...log, { seq: 3, kind: 'message', payload: { text: 'three' } }]
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(reads()).toEqual([0, 2])
    expect(screen.getAllByText(/^(one|two|three)$/).map(node => node.textContent)).toEqual(['one', 'two', 'three'])
    expect(request.mock.calls.filter(call => call[1] === 'groups.state')).toHaveLength(2)
    epoch = 2
    log = [{ seq: 1, kind: 'message', payload: { text: 'fresh' } }]
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(reads()).toEqual([0, 2, 0])
    expect(screen.queryByText('one')).toBeNull()
    expect(screen.getByText('fresh')).toBeTruthy()
    view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
    const before = request.mock.calls.length
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(request.mock.calls).toHaveLength(before)
  } finally {
    vi.useRealTimers()
  }
})

it('keeps Stop available while a Send is pending and reports a request without claiming completion', async () => {
  const stops: Record<string, unknown>[] = []
  let stopResult: () => Promise<unknown> = async () => ({ cancelled: 0 })
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { running: true, working: false } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return new Promise(() => {})}

    if (method === 'groups.stop') {
      stops.push(params)

      return stopResult()
    }

    return {}
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'stop-room' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Long task' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  const stop = screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await screen.findByText(labels.nothingRunning)
  expect(stops).toEqual([{ room_id: 'stop-room', cancel_id: expect.any(String), profile: 'team' }])

  stopResult = async () => { throw new Error('stop refused') }
  fireEvent.click(stop)
  expect((await screen.findByRole('alert')).textContent).toContain(labels.pendingActionUnconfirmed)
  expect(screen.getByText('stop refused').closest('details')?.open).toBe(false)
})

it('keeps an idle chat quiet while its gateway is alive, even with a draft and file upload', async () => {
  let releaseUpload!: (value: unknown) => void
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Autumn launch' }, driver_status: { running: true, working: false, counts: { completed: 2 } } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.attachment.upload') {return new Promise(resolve => { releaseUpload = resolve })}

    return {}
  })
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'idle-owner', profile: 'team', roomId: 'idle' }} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(input.disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  fireEvent.change(input, { target: { value: 'Review this file' } })
  fireEvent.change(view.container.querySelector('input[type=file]')!, { target: { files: [new File(['A'], 'notes.txt', { type: 'text/plain' })] } })
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.upload')).toBe(true))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true)
  await act(async () => releaseUpload({ attachment_id: 'uploaded', kind: 'file', name: 'notes.txt', mime: 'text/plain' }))
  await waitFor(() => expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
})

it.each([
  ['queued work', { working: false, counts: { queued: 1 } }],
  ['stopping work', { working: false, counts: { stopping: 1 } }],
  ['unresolved reply', { working: false, pending_actions: [{ kind: 'discard', member_id: 'worker', task_id: 'pending', execution_generation: 1 }] }]
])('offers Stop for %s from the current driver receipt', async (_description, driver_status) => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch' }, driver_status } : method === 'groups.log' ? { events: [] } : {})
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'active-owner', profile: 'team', roomId: _description }} />)
  const stop = await screen.findByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
})

it('keeps Stop usable for an unresolved frozen Send even when driver status is unavailable', async () => {
  const binding = { connectionId: 'unconfirmed-owner', profile: 'team', roomId: 'unconfirmed' }
  await prepareCanonicalGroupSend(binding, { text: 'Please review the launch' })
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch' } } : method === 'groups.log' ? { events: [] } : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  const stop = await screen.findByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
  expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
})

it('shows every current participant without a network action and closes the popover when hidden', async () => {
  const members = Array.from({ length: 4 }, (_, index) => ({ member_id: `member-${index}`, profile: `profile-${index}`,
    handle: `handle-${index}`, display_name: index % 2 ? 'Mira Bot' : 'Atlas Bot' }))

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch', members }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
  const binding = { connectionId: 'participant-owner', profile: 'team', roomId: 'participants' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const trigger = await screen.findByRole('button', { name: `${labels.members}: ${labels.memberCount.replace('{count}', '4')}` })
  const reads = request.mock.calls.length
  fireEvent.click(trigger)
  const list = within(await screen.findByRole('list', { name: labels.members }))
  expect(list.getAllByText('Atlas Bot')).toHaveLength(2)
  expect(list.getAllByText('Mira Bot')).toHaveLength(2)
  expect(list.queryByText('handle-3')).toBeNull()
  expect(request.mock.calls).toHaveLength(reads)
  view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
  expect(screen.queryByRole('list', { name: labels.members })).toBeNull()
})

it('lists an unresolved member beside live work without disabling Stop or polling', async () => {
  vi.useFakeTimers()

  try {
    request.mockImplementation(async (_route, method) => method === 'groups.state'
      ? { room: { name: 'Room' }, driver_status: { running: true, working: true,
        pending_actions: [{ kind: 'retry', member_id: 'alpha', task_id: 't', execution_generation: 1 }] } }
      : method === 'groups.log' ? { events: [] } : {})
    render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'warn' }} />)
    await settle()
    expect(screen.getByText(`${labels.statusWorking} · ${labels.statusAttention.replace('{count}', '1')}`)).toBeTruthy()
    expect(screen.getByText(labels.pendingRetryTitle.replace('{name}', labels.pendingBot))).toBeTruthy()
    expect(screen.queryByText('alpha')).toBeNull()
    expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
    const polls = request.mock.calls.filter(call => call[1] === 'groups.state').length
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(request.mock.calls.filter(call => call[1] === 'groups.state').length).toBeGreaterThan(polls)
  } finally {
    vi.useRealTimers()
  }
})

const refusal = (reason: string) => Object.assign(new Error(reason), { code: 4001, data: { reason } })

it('hands a refused message back for editing and keeps other outcomes for an exact retry', async () => {
  let outcome: () => Promise<unknown> = async () => { throw refusal('invalid_params') }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return outcome()}

    return {}
  })
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'outcomes' }
  render(<CanonicalGroupWorkspace binding={binding} />)
  const box = () => screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box().disabled).toBe(false))
  fireEvent.change(box(), { target: { value: 'Too long' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(labels.sendRefused)
  await waitFor(() => expect(box().disabled).toBe(false))
  expect(box().value).toBe('Too long')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()

  outcome = async () => { throw refusal('not_ready') }
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(labels.sendNotYet)
  const kept = await readCanonicalGroupSend(binding)
  expect(kept?.params.payload.text).toBe('Too long')

  outcome = async () => { throw new Error('socket closed') }
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await screen.findByText(labels.sendMaybe)
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(kept?.params.event_id)
  const ids = request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2].event_id)
  expect(ids[0]).not.toBe(ids[1])
  expect(ids[2]).toBe(ids[1])

  outcome = async () => ({ accepted: true, client_event_id: 'someone-else' })
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.send')).toHaveLength(4))
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(kept?.params.event_id)

  outcome = async () => ({ accepted: true, client_event_id: kept?.params.event_id, driver_started: true })
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  await waitFor(() => expect(box().value).toBe(''))
})

it('returns a failed Send only to its own room when the view switches rooms', async () => {
  let refuse!: () => void
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return new Promise((_resolve, reject) => { refuse = () => reject(refusal('invalid_params')) })}

    return {}
  })
  const first = { connectionId: 'remote', profile: 'team', roomId: 'room-a' }
  const second = { connectionId: 'remote', profile: 'team', roomId: 'room-b' }
  const view = render(<CanonicalGroupWorkspace binding={first} />)
  const box = () => screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box().disabled).toBe(false))
  fireEvent.change(box(), { target: { value: 'For room A' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  view.rerender(<CanonicalGroupWorkspace binding={second} />)
  await waitFor(() => expect(box().disabled).toBe(false))
  refuse()
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })
  expect(box().value).toBe('')
  expect(screen.queryByRole('alert')).toBeNull()
  expect((await readCanonicalGroupSend(first))?.params.payload.text).toBe('For room A')
  view.rerender(<CanonicalGroupWorkspace binding={first} />)
  await waitFor(() => expect(box().value).toBe('For room A'))
})

it('renames a gateway room with one event id across retries', async () => {
  let name = 'Old name'

  let renameOutcome: () => Promise<unknown> = async () => { throw new Error('socket closed') }

  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.rename') {
      const result = await renameOutcome()
      name = params.name

      return result
    }

    return {}
  })
  const group = registerCanonicalGroup({ connectionId: 'rename-owner', profile: 'team' }, { room_id: 'named', name: 'Old name', members: [] })
  render(<GroupChatWorkspace group={group} members={[]} />)
  await chooseGroupAction(labels.rename)
  fireEvent.change(screen.getByRole('textbox', { name: labels.roomName }), { target: { value: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toContain(labels.pendingActionUnconfirmed)
  expect(screen.getByText('socket closed').closest('details')?.open).toBe(false)

  renameOutcome = async () => ({ room: { room_id: 'named', name: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await screen.findByRole('heading', { name: 'New name' })
  expect($canonicalGroupNames.get()[group]).toBe('New name')
  const renames = request.mock.calls.filter(call => call[1] === 'groups.rename').map(call => call[2])
  expect(renames).toEqual([
    { room_id: 'named', event_id: expect.any(String), name: 'New name', profile: 'team' },
    { room_id: 'named', event_id: renames[0].event_id, name: 'New name', profile: 'team' }
  ])
})

it.each([
  ['missing receipt', {}],
  ['legacy boolean', { tombstone: true }],
  ['another room', { tombstone: { room_id: 'other-room', disbanded_at: 123, idempotent: false } }],
  ['missing timestamp', { tombstone: { room_id: 'leaving', idempotent: false } }]
])('keeps a room for %s and accepts its canonical Disband receipt', async (_label, initial) => {
  let disbandResult: unknown = initial
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name: 'Leaving' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.disband') {return disbandResult}

    return {}
  })
  const binding = { connectionId: 'disband-owner', profile: 'team', roomId: 'leaving' }
  const key = registerCanonicalGroup(binding, { room_id: 'leaving', name: 'Leaving', members: [] })
  const onBack = vi.fn()
  render(<GroupChatWorkspace group={key} members={[]} onBack={onBack} />)
  await chooseGroupAction(labels.disband)
  expect(request.mock.calls.some(call => call[1] === 'groups.disband')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDisband }))
  await screen.findByText(labels.disbandUnconfirmed)
  expect($canonicalGroupBindings.get()[key]).toEqual(binding)
  expect(onBack).not.toHaveBeenCalled()

  disbandResult = { tombstone: { room_id: binding.roomId, disbanded_at: 123, idempotent: false } }
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDisband }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
  expect($canonicalGroupBindings.get()[key]).toBeUndefined()
  expect(request.mock.calls.filter(call => call[1] === 'groups.disband').map(call => call[2])).toEqual([
    { room_id: 'leaving', cancel_id: expect.any(String), profile: 'team' },
    { room_id: 'leaving', cancel_id: request.mock.calls.find(call => call[1] === 'groups.disband')![2].cancel_id, profile: 'team' }
  ])
})

it('shows shared confirmation progress and prevents duplicate End requests while its receipt is pending', async () => {
  let release!: (value: unknown) => void
  const held = new Promise(resolve => { release = resolve })
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name: 'Autumn launch' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.disband') {return held}

    return {}
  })
  const binding = { connectionId: 'end-busy-owner', profile: 'team', roomId: 'ending' }
  const key = registerCanonicalGroup(binding, { room_id: binding.roomId, name: 'Autumn launch', members: [] })
  const onBack = vi.fn()
  render(<GroupChatWorkspace group={key} members={[]} onBack={onBack} />)
  await chooseGroupAction(labels.disband)
  const dialog = within(screen.getByRole('dialog'))
  const confirm = dialog.getByRole('button', { name: labels.confirmDisband }) as HTMLButtonElement
  fireEvent.click(confirm)
  await waitFor(() => expect(confirm.disabled).toBe(true))
  expect((dialog.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(confirm)
  expect(request.mock.calls.filter(call => call[1] === 'groups.disband')).toHaveLength(1)
  expect(onBack).not.toHaveBeenCalled()
  await act(async () => release({ tombstone: { room_id: binding.roomId, disbanded_at: 123 } }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
})


it('shows ordinary file publication as work while preserving attention for blocked or uncertain actions', async () => {
  const output = { kind: 'output_retry', operation: 'ack', blocked: false, member_id: 'atlas', task_id: 'files', execution_generation: 1 }
  let driver_status = { running: true, working: false, blocked: true, pending_actions: [output] }
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Launch', members: [{ member_id: 'atlas', profile: 'default', display_name: 'Atlas Bot' }] }, driver_status }
    : method === 'groups.log' ? { events: [] } : {})
  const binding = { connectionId: 'home', profile: 'default', roomId: 'file-sharing' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('Sharing files from Atlas Bot…')
  expect(screen.getByText(labels.statusWorking)).toBeTruthy()
  expect(screen.queryByText(new RegExp(labels.statusBlocked))).toBeNull()
  driver_status = { ...driver_status, pending_actions: [{ ...output, blocked: true }] }
  view.rerender(<CanonicalGroupWorkspace binding={{ ...binding, roomId: 'file-blocked' }} />)
  await screen.findByText('Files from Atlas Bot need attention.')
  expect(screen.getByText(new RegExp(labels.statusBlocked))).toBeTruthy()
  driver_status = { ...driver_status, pending_actions: [output, { ...output, kind: 'discard', task_id: 'uncertain' }] }
  view.rerender(<CanonicalGroupWorkspace binding={{ ...binding, roomId: 'file-and-uncertain' }} />)
  await screen.findByText('Sharing files from Atlas Bot…')
  expect(screen.getByText(new RegExp(labels.statusBlocked))).toBeTruthy()
})

it('keeps its frozen send for missing or malformed acknowledgement identity', async () => {
  const binding = { connectionId: 'local', profile: 'default', roomId: 'receipt' }
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Exact intent' })
  let receipt: unknown = { accepted: true }
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room' }, driver_status: {} }
      : method === 'groups.log'
        ? { events: [] }
        : method === 'groups.send'
          ? receipt
          : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(false))
  for (const value of [{ accepted: true }, { accepted: true, client_event_id: 1 }, { client_event_id: 'wrong' }]) {
    receipt = value
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES.en.unconfirmedSend)
    )
    expect(await readCanonicalGroupSend(binding)).toEqual({ ...entry, attempted: true })
    await waitFor(() =>
      expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(false)
    )
  }
  receipt = { accepted: true, client_event_id: entry.params.event_id }
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
})

it('coalesces rapid Stop gestures and reuses the same cancel intent after a lost reply', async () => {
  let reject!: (error: Error) => void
  let complete!: (value: unknown) => void
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room' }, driver_status: { working: true } }
      : method === 'groups.log'
        ? { events: [] }
        : method === 'groups.stop'
          ? new Promise((resolve, fail) => {
      complete = resolve
              reject = fail
            })
          : {}
  )
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: 'rapid-stop' }} />)
  const stop = await screen.findByRole('button', { name: 'Stop' })
  await act(async () => {
    fireEvent.click(stop)
    fireEvent.click(stop)
  })
  expect(request.mock.calls.filter(call => call[1] === 'groups.stop')).toHaveLength(1)
  await act(async () => {
    reject(new Error('lost stop reply'))
  })
  const original = request.mock.calls.find(call => call[1] === 'groups.stop')![2].cancel_id
  await act(async () => {
    fireEvent.click(stop)
  })
  expect(request.mock.calls.filter(call => call[1] === 'groups.stop').map(call => call[2].cancel_id)).toEqual([
    original,
    original
  ])
  await act(async () => {complete({})})
  expect(screen.queryByText(CANONICAL_GROUP_LOCALES.en.nothingRunning)).toBeNull()
  expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES.en.pendingActionUnconfirmed)
  await act(async () => {fireEvent.click(stop)})
  expect(request.mock.calls.filter(call => call[1] === 'groups.stop').map(call => call[2].cancel_id)).toEqual([original, original, original])
  await act(async () => {complete({cancelled: 0})})
  expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.nothingRunning)).toBeTruthy()
})

it('does not offer Stop for file cleanup alone but keeps it for unknown execution', async () => {
  let driver_status: {
    running: boolean
    working: boolean
    counts: Record<string, number>
    pending_actions: Array<{
      kind: string
      operation: string
      blocked: boolean
      member_id: string
      task_id: string
      execution_generation: number
    }>
  } = {
    running: true,
    working: false,
    counts: { settled: 1 },
    pending_actions: [
      {
        kind: 'output_retry',
        operation: 'discard',
        blocked: true,
        member_id: 'atlas',
        task_id: 'done-files',
        execution_generation: 1
      }
    ]
  }
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room' }, driver_status }
      : method === 'groups.log'
        ? { events: [] }
        : {}
  )
  const binding = { connectionId: 'local', profile: 'default', roomId: 'cleanup-stop' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText(/Files from .* need attention/)
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  view.unmount()
  driver_status = {
    running: true,
    working: false,
    counts: { unknown: 1 },
    pending_actions: [
      {
        kind: 'unknown',
        operation: '',
        blocked: false,
        member_id: 'atlas',
        task_id: 'unknown-task',
        execution_generation: 1
      }
    ]
  }
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByRole('button', { name: 'Stop' })).toBeTruthy()
})
