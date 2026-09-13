import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { claimPreparedSubmission, preparedSubmissionKey, readPreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
import { captureSubmissionDestination } from './submission-destination'
import { useSubmitPrompt } from './submit'
import { clearSubmitInFlight, type GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => {})
}))

let restoreNative: typeof window.hermesDesktop
let home: string

beforeEach(() => {
  restoreNative = window.hermesDesktop
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'send-journal-'))
  localStorage.clear()
  clearSubmitInFlight()
  $connection.set(null)
  $sessions.set([])
  $sessionStates.set({})
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  clearSubmitInFlight()
  window.hermesDesktop = restoreNative
  fs.rmSync(home, { recursive: true, force: true })
  vi.doUnmock('electron')
  vi.restoreAllMocks()
})

async function storeFor(owner: string) {
  vi.doMock('electron', () => ({ app: {}, ipcMain: {} }))
  const nativeModule = '../../../../../electron/prepared-submissions'
  const { preparedJournal } = await import(nativeModule)
  const store = preparedJournal(home, 'http://same-renderer-origin')

  const bridge = {
    owner: async () => owner,
    read: async () => JSON.stringify(store.read()),
    update: async (key: string, entry: string | null) => store.update(key, entry === null ? null : JSON.parse(entry)),
    compareAndSet: vi.fn(async (key: string, expected: string | null, entry: string | null) =>
      store.compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry)))
  }

  return { store, bridge, bind: () => {window.hermesDesktop = { ...restoreNative, preparedSubmissions: bridge }} }
}

function composer(request: GatewayRequest) {
  let state = createClientSessionState()

  const deps = {
    activeSessionIdRef: { current: 'runtime' as string | null },
    selectedStoredSessionIdRef: { current: 'stored' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime',
    getRouteToken: () => 'same-route',
    requestGateway: request,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored', 'runtime']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: (_sid: string, update: (state: ClientSessionState) => ClientSessionState) => { state = update(state);

 return state }
  }

  const hook = renderHook(() => useSubmitPrompt(deps))

  return { hook, state: () => state }
}

it('keeps independent window intents and reuses only the original window retry', async () => {
  const a = await storeFor('window-a')
  const b = await storeFor('window-b')
  const submissions: Record<string, unknown>[] = []
  let refuse = true

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method !== 'prompt.submit') {return {} as never}
    submissions.push(params!)

    if (refuse) {throw new Error('connection closed')}

    return { admission_id: params?.submission_id, status: 'started' } as never
  })

  a.bind()
  let view = composer(request)
  await act(async () => {expect(await view.hook.result.current('same text')).toBe(false)})
  view.hook.unmount()
  b.bind()
  refuse = false
  view = composer(request)
  await act(async () => {expect(await view.hook.result.current('same text')).toBe(true)})
  view.hook.unmount()
  expect(submissions[1].submission_id).not.toBe(submissions[0].submission_id)
  expect(Object.values(a.store.read())).toEqual([expect.objectContaining({ id: submissions[0].submission_id })])
  a.bind()
  view = composer(request)
  await act(async () => {expect(await view.hook.result.current('same text')).toBe(true)})
  expect(submissions[2]).toEqual(submissions[0])
  expect(a.store.read()).toEqual({})
})

it('atomically transfers an explicitly chosen draft without retargeting or letting the prior owner delete it', async () => {
  const a = await storeFor('window-a')
  const b = await storeFor('window-b')
  const request = vi.fn() as GatewayRequest
  const destination = captureSubmissionDestination('stored', request)
  const key = preparedSubmissionKey('stored', destination, 'same text', [])
  a.bind()
  const first = { id: 'intent-a', text: 'same text', attachments: [], owner: destination.owner, params: { session_id: 'runtime', submission_id: 'intent-a' } }
  await writePreparedSubmission(key, first)
  const retainedA = (await readPreparedSubmission(key))!
  b.bind()
  expect(await readPreparedSubmission(key)).toBeUndefined()
  await writePreparedSubmission(key, { ...first, id: 'intent-b', journal: undefined, params: { ...first.params, submission_id: 'intent-b' } })
  await claimPreparedSubmission(retainedA.journal!.storageKey)
  const recovered = (await readPreparedSubmission(key))!
  expect(recovered.id).toBe('intent-a')
  expect(recovered.params).toEqual(first.params)
  a.bind()
  await expect(removePreparedSubmission(key, retainedA)).rejects.toThrow('changed before retirement')
  expect(Object.values(a.store.read()).map((entry: any) => entry.id).sort()).toEqual(['intent-a', 'intent-b'])
  b.bind()
  await removePreparedSubmission(key, recovered)
  expect(Object.values(b.store.read()).map((entry: any) => entry.id)).toEqual(['intent-b'])
})

it.each(['queued', 'started', 'terminal'])('keeps a valid %s ACK successful when removal fails and never resubmits its explicit retry', async status => {
  const a = await storeFor('window-a')
  a.bind()
  const compare = a.bridge.compareAndSet.getMockImplementation()!
  a.bridge.compareAndSet.mockImplementation(async (key, expected, entry) => {
    if (entry === null) {throw new Error('fixture remove EACCES')}

    return compare(key, expected, entry)
  })

  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => ({
    admission_id: 'canonical-admission', submission_id: params?.submission_id, session_id: params?.session_id, status
  }) as never)

  const view = composer(request)
  const options = { submission_id: 'accepted-id' }
  await act(async () => {expect(await view.hook.result.current('accepted text', options)).toBe(true)})
  expect(view.state().messages.filter(message => message.error)).toEqual([])
  expect(view.state().busy).toBe(status !== 'terminal')
  expect(Object.values(a.store.read())).toEqual([expect.objectContaining({ id: 'accepted-id', acknowledged: true })])
  await act(async () => {expect(await view.hook.result.current('accepted text', options)).toBe(true)})
  expect(request).toHaveBeenCalledOnce()
})

it('keeps accepted identity in memory even when both ACK marking and removal fail', async () => {
  const a = await storeFor('window-a')
  a.bind()
  const compare = a.bridge.compareAndSet.getMockImplementation()!
  a.bridge.compareAndSet.mockImplementation(async (key, expected, entry) => {
    if (expected !== null) {throw new Error('fixture disk unavailable after ACK')}

    return compare(key, expected, entry)
  })
  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => ({ admission_id: params?.submission_id, status: 'started' }) as never)
  const view = composer(request)
  const options = { submission_id: 'memory-accepted' }
  await act(async () => {expect(await view.hook.result.current('accepted', options)).toBe(true)})
  await act(async () => {expect(await view.hook.result.current('accepted', options)).toBe(true)})
  expect(request).toHaveBeenCalledOnce()
  expect(view.state().messages.filter(message => message.error)).toEqual([])
  expect(Object.values(a.store.read())).toEqual([expect.objectContaining({ id: 'memory-accepted' })])
})

it('gives a later independent Send a new ID even when the preceding accepted journal could not be removed', async () => {
  const a = await storeFor('window-a')
  a.bind()
  const compare = a.bridge.compareAndSet.getMockImplementation()!
  a.bridge.compareAndSet.mockImplementation(async (key, expected, entry) => {
    if (entry === null) {throw new Error('fixture remove EACCES')}

    return compare(key, expected, entry)
  })
  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => ({ admission_id: params?.submission_id, status: 'terminal' }) as never)
  const view = composer(request)
  await act(async () => {expect(await view.hook.result.current('same text')).toBe(true)})
  await act(async () => {expect(await view.hook.result.current('same text')).toBe(true)})
  expect(request.mock.calls[0][1]?.submission_id).not.toBe(request.mock.calls[1][1]?.submission_id)
})
