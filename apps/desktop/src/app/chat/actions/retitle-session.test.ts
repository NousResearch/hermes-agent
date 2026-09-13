import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  activeGateway: vi.fn(),
  activeSessionId: { get: vi.fn() },
  notify: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn(),
  selectedStoredSessionId: { get: vi.fn() },
  sessions: { get: vi.fn() },
  setSessions: vi.fn()
}))

vi.mock('@/i18n', () => ({
  translateNow: () => ({
    sidebar: {
      row: {
        regenerateTitleFailed: 'Could not regenerate session title',
        regenerateTitleSuccess: 'Session title regenerated',
        regeneratingTitle: 'Regenerating session title...'
      }
    }
  })
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: mocks.activeGateway
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('@/store/session', () => ({
  $activeSessionId: mocks.activeSessionId,
  $selectedStoredSessionId: mocks.selectedStoredSessionId,
  $sessions: mocks.sessions,
  sessionMatchesStoredId: (session: { _lineage_root_id?: string; id: string }, storedId: string) =>
    session.id === storedId || session._lineage_root_id === storedId,
  setSessions: mocks.setSessions
}))

import { runSessionRetitle, sessionRetitleMatchesSelection } from './retitle-session'

describe('runSessionRetitle', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.activeSessionId.get.mockReturnValue('runtime-1')
    mocks.selectedStoredSessionId.get.mockReturnValue('stored-1')
    mocks.sessions.get.mockReturnValue([{ id: 'stored-1', title: 'Old title' }])
    mocks.activeGateway.mockReturnValue({ request: mocks.request })
  })

  it('calls session.retitle for the active runtime session and applies the returned stored title', async () => {
    mocks.request.mockResolvedValue({ title: 'New title' })

    await expect(runSessionRetitle({ sessionId: 'stored-1' })).resolves.toBe('New title')

    expect(mocks.request).toHaveBeenCalledWith('session.retitle', { session_id: 'runtime-1' })
    expect(mocks.setSessions).toHaveBeenCalledTimes(1)

    const update = mocks.setSessions.mock.calls[0][0]
    const before = [
      { id: 'stored-1', title: 'Old title' },
      { id: 'stored-2', title: 'Other title' }
    ]
    expect(update(before)).toEqual([
      { id: 'stored-1', title: 'New title' },
      { id: 'stored-2', title: 'Other title' }
    ])
    expect(mocks.notify).toHaveBeenLastCalledWith({
      durationMs: 2_000,
      kind: 'success',
      message: 'Session title regenerated'
    })
  })

  it('treats a compression tip and its selected lineage root as the same session', async () => {
    mocks.selectedStoredSessionId.get.mockReturnValue('root-1')
    mocks.sessions.get.mockReturnValue([{ id: 'tip-2', _lineage_root_id: 'root-1', title: 'Old title' }])
    mocks.request.mockResolvedValue({ title: 'Compressed conversation' })

    expect(sessionRetitleMatchesSelection('tip-2')).toBe(true)
    await expect(runSessionRetitle({ sessionId: 'tip-2' })).resolves.toBe('Compressed conversation')

    expect(mocks.request).toHaveBeenCalledWith('session.retitle', { session_id: 'runtime-1' })
  })

  it('does nothing if the menu row is no longer the selected session', async () => {
    await expect(runSessionRetitle({ sessionId: 'stored-2' })).resolves.toBeNull()

    expect(mocks.request).not.toHaveBeenCalled()
    expect(mocks.setSessions).not.toHaveBeenCalled()
  })

  it('surfaces backend errors without mutating the sidebar title', async () => {
    const error = new Error('retitle failed')
    mocks.request.mockRejectedValue(error)

    await expect(runSessionRetitle({ sessionId: 'stored-1' })).resolves.toBeNull()

    expect(mocks.setSessions).not.toHaveBeenCalled()
    expect(mocks.notifyError).toHaveBeenCalledWith(error, 'Could not regenerate session title')
  })
})
