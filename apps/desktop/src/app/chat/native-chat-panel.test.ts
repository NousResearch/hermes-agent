import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const {
  bindCreatedSession,
  requestGatewayForAgent,
  retainGatewayForAgent,
  sessionTileDelegate,
  setSessionOwnerHint
} = vi.hoisted(() => ({
  bindCreatedSession: vi.fn(),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForAgent: vi.fn(),
  sessionTileDelegate: vi.fn(),
  setSessionOwnerHint: vi.fn()
}))

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent, retainGatewayForAgent }))
vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<typeof import('@/store/session')>()),
  setSessionOwnerHint
}))
vi.mock('@/store/session-states', () => ({
  $sessionStates: atom({}),
  $sessionTileDelegateRevision: atom(1),
  retainForegroundSessionSurface: vi.fn(() => () => undefined),
  sessionTileDelegate
}))
vi.mock('./composer/focus', () => ({ requestComposerFocus: vi.fn() }))

import {
  _nativeChatAttachmentScopeForTests,
  _resetNativeChatSessionLeasesForTests,
  createNativeChatSession
} from './native-chat-panel'

const route = {
  connectionId: 'scope-internal',
  mode: 'remote' as const,
  profile: 'Internal',
  targetProfile: 'internal-workspace'
}

describe('createNativeChatSession', () => {
  beforeEach(() => {
    requestGatewayForAgent.mockReset()
    retainGatewayForAgent.mockReset()
    setSessionOwnerHint.mockReset()
    bindCreatedSession.mockReset()
    sessionTileDelegate.mockReset()
    sessionTileDelegate.mockReturnValue({ bindCreatedSession })
  })

  afterEach(() => {
    _resetNativeChatSessionLeasesForTests()
  })

  it('creates and binds on the exact profile route without navigation state', async () => {
    const release = vi.fn()
    retainGatewayForAgent.mockResolvedValue(release)
    requestGatewayForAgent.mockResolvedValue({
      info: { model: 'internal-model' },
      session_id: 'runtime-1',
      stored_session_id: 'stored-1'
    })
    bindCreatedSession.mockReturnValue('runtime-1')

    await expect(createNativeChatSession({ route })).resolves.toEqual({
      route,
      runtimeSessionId: 'runtime-1',
      storedSessionId: 'stored-1'
    })

    expect(retainGatewayForAgent).toHaveBeenCalledWith('scope-internal', 'Internal')
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'scope-internal',
      'Internal',
      'session.create',
      expect.objectContaining({ hidden: true, profile: 'internal-workspace', source: 'desktop' })
    )
    expect(setSessionOwnerHint).toHaveBeenCalledWith('stored-1', route)
    expect(bindCreatedSession).toHaveBeenCalledWith(expect.objectContaining({ session_id: 'runtime-1' }), 'stored-1')
    expect(release).not.toHaveBeenCalled()
  })

  it('closes the runtime and releases the route when creation cannot be durably bound', async () => {
    const release = vi.fn()
    retainGatewayForAgent.mockResolvedValue(release)
    requestGatewayForAgent
      .mockResolvedValueOnce({ session_id: 'runtime-orphan' })
      .mockResolvedValueOnce({ status: 'closed' })

    await expect(createNativeChatSession({ route })).rejects.toThrow('without a stored session id')

    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(2, 'scope-internal', 'Internal', 'session.close', {
      session_id: 'runtime-orphan'
    })
    expect(bindCreatedSession).not.toHaveBeenCalled()
    expect(release).toHaveBeenCalledTimes(1)
  })

  it('rejects an incomplete ambient route before opening any gateway', async () => {
    await expect(
      createNativeChatSession({ route: { ...route, targetProfile: '' } })
    ).rejects.toThrow('exact connectionId + mode + profile + targetProfile')

    expect(retainGatewayForAgent).not.toHaveBeenCalled()
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
  })

  it('retains unsent attachments for the same owner-qualified native binding only', () => {
    const first = _nativeChatAttachmentScopeForTests(route, 'stored-attachments')
    first.add({
      id: 'file-1',
      kind: 'file',
      label: 'quote.pdf',
      occurrenceId: 'occurrence-1',
      path: '/workspace/quote.pdf'
    })

    const reopened = _nativeChatAttachmentScopeForTests(route, 'stored-attachments')
    const otherOwner = _nativeChatAttachmentScopeForTests(
      { ...route, connectionId: 'scope-other' },
      'stored-attachments'
    )

    expect(reopened).toBe(first)
    expect(reopened.$attachments.get()).toHaveLength(1)
    expect(reopened.$attachments.get()[0]).toMatchObject({ id: 'file-1', label: 'quote.pdf' })
    expect(otherOwner).not.toBe(first)
    expect(otherOwner.$attachments.get()).toEqual([])
  })
})
