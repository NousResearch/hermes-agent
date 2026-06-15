import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  requestGatewayForEndpoint: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  sessions: [] as SessionInfo[]
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string, count: number) => `${key}:${count}`
}))

vi.mock('@/store/gateway', () => ({
  requestGatewayForEndpoint: mocks.requestGatewayForEndpoint,
  requestGatewayForProfile: mocks.requestGatewayForProfile
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('@/store/profile', () => ({
  normalizeProfileKey: (profile?: null | string) => profile || 'default'
}))

vi.mock('@/store/remote-sessions', () => ({
  remoteSessionEndpoint: () => null
}))

vi.mock('@/store/session', () => ({
  $archivedSessions: { get: () => [] },
  $cronSessions: { get: () => [] },
  $localDeviceName: { get: () => 'ko-mac' },
  $messagingSessions: { get: () => [] },
  $sessions: { get: () => mocks.sessions }
}))

import { promptStoredSessions } from './session-bulk-runtime-actions'

function session(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    _lineage_root_id: null,
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 1000,
    title: 'Session',
    tool_call_count: 0,
    id: 's1',
    ...overrides
  } as SessionInfo
}

describe('session bulk runtime actions', () => {
  beforeEach(() => {
    mocks.notify.mockReset()
    mocks.notifyError.mockReset()
    mocks.requestGatewayForEndpoint.mockReset()
    mocks.requestGatewayForProfile.mockReset()
    mocks.sessions = [session()]
  })

  it('queues a bulk prompt when direct submit reports a busy session', async () => {
    const calls: Array<{ method: string; params: Record<string, unknown>; profile: string }> = []

    mocks.requestGatewayForProfile.mockImplementation(async (profile, method, params = {}) => {
      calls.push({ method, params, profile })

      if (method === 'session.resume') {
        return { session_id: 'runtime-s1' }
      }

      if (method === 'prompt.submit') {
        throw new Error('session busy')
      }

      if (method === 'prompt.queue') {
        return { depth: 1, status: 'queued' }
      }

      return {}
    })

    const result = await promptStoredSessions(['s1'], '  keep going  ')

    expect(result).toEqual({ failed: [], ok: ['s1'] })
    expect(calls).toEqual([
      {
        method: 'session.resume',
        params: { cols: 96, session_id: 's1', viewer_device: 'ko-mac', profile: 'default' },
        profile: 'default'
      },
      {
        method: 'prompt.submit',
        params: { session_id: 'runtime-s1', text: 'keep going' },
        profile: 'default'
      },
      {
        method: 'prompt.queue',
        params: { session_id: 'runtime-s1', text: 'keep going' },
        profile: 'default'
      }
    ])
    expect(mocks.notify).toHaveBeenCalledWith({
      durationMs: 2500,
      kind: 'success',
      message: 'sidebar.bulk.promptedToast:1'
    })
    expect(mocks.notifyError).not.toHaveBeenCalled()
  })

  it('does not hide non-busy prompt submit failures behind the queue', async () => {
    const methods: string[] = []

    mocks.requestGatewayForProfile.mockImplementation(async (_profile, method) => {
      methods.push(method)

      if (method === 'session.resume') {
        return { session_id: 'runtime-s1' }
      }

      throw new Error('gateway unavailable')
    })

    const result = await promptStoredSessions(['s1'], 'hello')

    expect(result.ok).toEqual([])
    expect(result.failed).toHaveLength(1)
    expect(methods).toEqual(['session.resume', 'prompt.submit'])
    expect(mocks.notifyError).toHaveBeenCalledTimes(1)
  })
})
