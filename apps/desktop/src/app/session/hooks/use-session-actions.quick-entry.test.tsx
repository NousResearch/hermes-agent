import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $newChatProfile, $newChatRoute } from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { $projectTree } from '@/store/projects'
import { $currentCwdExplicit, setSessions } from '@/store/session'

import type { ClientSessionState } from '../../types'

import { pinnedOwnerCount } from './session-context-drift'
import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => {
  const original = await importOriginal<Record<string, unknown>>()

  return {
    ...original,
    activeGatewayConnectionId: vi.fn(original.activeGatewayConnectionId as () => null | string),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => () => undefined)
  }
})

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

type SubmitTextToNewSession = ReturnType<typeof useSessionActions>['submitTextToNewSession']

function QuickEntryHarness({
  onReady,
  requestGateway
}: {
  onReady: (submit: SubmitTextToNewSession) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<null | string>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<null | string>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions.submitTextToNewSession)
  }, [actions, onReady])

  return null
}

describe('Quick Entry gateway compatibility', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $currentCwdExplicit.set(false)
    setSessions([])
    vi.clearAllMocks()
  })

  it.each([true, false])('creates one Quick Entry session and submits once (older gateway: %s)', async older => {
    const cwd = '/work/project'
    $newChatProfile.set('docker')
    $newChatRoute.set(null)
    $projectScope.set('quick-project')
    $projectTree.set([{ id: 'quick-project', label: 'Project', path: cwd, repos: [], sessionCount: 0 }])
    $currentCwdExplicit.set(true)

    const acceptedCreates: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        // Older strict contracts reject at admission, before any session exists.
        if (older && params && 'cwd_explicit' in params) {
          throw new JsonRpcGatewayError(
            'invalid params for session.create: cwd_explicit: Extra inputs are not permitted',
            { code: 4000 }
          )
        }

        acceptedCreates.push(params ?? {})

        return { session_id: 'runtime-quick', stored_session_id: 'stored-quick' } as never
      }

      return {} as never
    })

    let submit: SubmitTextToNewSession | null = null
    render(<QuickEntryHarness onReady={value => (submit = value)} requestGateway={requestGateway} />)

    await act(async () => {
      await expect(submit!('quick entry text')).resolves.toEqual({
        runtimeSessionId: 'runtime-quick',
        sessionId: 'stored-quick'
      })
    })

    expect(acceptedCreates).toHaveLength(1)
    expect(acceptedCreates[0]).toMatchObject({ cwd, profile: 'docker' })
    const attempts = requestGateway.mock.calls.filter(([method]) => method === 'session.create')
    expect(attempts[0][1]).toMatchObject({ cwd_explicit: true })

    if (older) {
      const { cwd_explicit: _flag, ...compatible } = attempts[0][1]!
      expect(attempts.map(([, params]) => params)).toEqual([attempts[0][1], compatible])
    } else {
      expect(attempts).toHaveLength(1)
      expect(acceptedCreates[0]).toHaveProperty('cwd_explicit', true)
    }

    expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-quick', text: 'quick entry text' }]
    ])
    expect(pinnedOwnerCount()).toBe(0)
  })
})
