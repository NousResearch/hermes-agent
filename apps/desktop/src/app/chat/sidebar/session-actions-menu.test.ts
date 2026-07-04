import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { renameSessionPreferringRpc } from './session-actions-menu'

// The branched-session rename bug: a freshly branched session lives only in the
// gateway's runtime _sessions map (no state.db row yet), so REST PATCH
// /api/sessions/{id} 404s with "Session not found". renameSessionPreferringRpc
// must route the ACTIVE row through the session.title RPC (runtime id), which
// persists the row on demand, and otherwise fall back to REST.

const renameSession = vi.fn(async () => ({ ok: true, title: 'rest-title' }))
const request = vi.fn(async () => ({ title: 'rpc-title' }) as never)
const activeGateway = vi.fn<() => { request: typeof request } | null>(() => ({ request }))

vi.mock('@/hermes', () => ({
  renameSession: (...args: unknown[]) => renameSession(...(args as [])),
  setApiRequestProfile: vi.fn(),
  HermesGateway: class {}
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: () => activeGateway()
}))

const RUNTIME_ID = 'rt-runtime-1'
const STORED_ID = 'stored-branch-1'

afterEach(() => {
  renameSession.mockClear()
  request.mockClear()
  activeGateway.mockReset()
  activeGateway.mockReturnValue({ request })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
})

describe('renameSessionPreferringRpc', () => {
  it('renames the active branched session via the session.title RPC, not REST', async () => {
    $selectedStoredSessionId.set(STORED_ID)
    $activeSessionId.set(RUNTIME_ID)

    const result = await renameSessionPreferringRpc(STORED_ID, 'My branch')

    expect(request).toHaveBeenCalledWith('session.title', { session_id: RUNTIME_ID, title: 'My branch' })
    expect(renameSession).not.toHaveBeenCalled()
    expect(result.title).toBe('rpc-title')
  })

  it('falls back to REST when the RPC fails (e.g. socket mid-reconnect)', async () => {
    $selectedStoredSessionId.set(STORED_ID)
    $activeSessionId.set(RUNTIME_ID)
    request.mockRejectedValueOnce(new Error('not connected'))

    const result = await renameSessionPreferringRpc(STORED_ID, 'My branch', 'work')

    expect(request).toHaveBeenCalledOnce()
    expect(renameSession).toHaveBeenCalledWith(STORED_ID, 'My branch', 'work')
    expect(result.title).toBe('rest-title')
  })

  it('uses REST for a non-active row (background/persisted session)', async () => {
    $selectedStoredSessionId.set('some-other-active-session')
    $activeSessionId.set(RUNTIME_ID)

    await renameSessionPreferringRpc(STORED_ID, 'My branch', 'work')

    expect(request).not.toHaveBeenCalled()
    expect(renameSession).toHaveBeenCalledWith(STORED_ID, 'My branch', 'work')
  })

  it('uses REST when clearing the title (RPC rejects empty titles)', async () => {
    $selectedStoredSessionId.set(STORED_ID)
    $activeSessionId.set(RUNTIME_ID)

    await renameSessionPreferringRpc(STORED_ID, '')

    expect(request).not.toHaveBeenCalled()
    expect(renameSession).toHaveBeenCalledWith(STORED_ID, '', undefined)
  })

  it('uses REST when no gateway is connected', async () => {
    $selectedStoredSessionId.set(STORED_ID)
    $activeSessionId.set(RUNTIME_ID)
    activeGateway.mockReturnValue(null)

    await renameSessionPreferringRpc(STORED_ID, 'My branch')

    expect(request).not.toHaveBeenCalled()
    expect(renameSession).toHaveBeenCalledWith(STORED_ID, 'My branch', undefined)
  })
})
