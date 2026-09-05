import { registryBackendScopeKey } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { getLatestSessionMessages, getSession } from '@/hermes'
import { useI18n } from '@/i18n'
import { host } from '@/sdk'
import type { AgentRow } from '@/store/agent-overview'
import { requestGatewayForAgent } from '@/store/gateway'
import { $gatewaySwapTarget, $showAllProfiles } from '@/store/profile'
import {
  $activeSessionId,
  $messages,
  $resumeExhaustedSessionId,
  $selectedStoredSessionId,
  $sessionResumeRequest,
  getSessionOwnerHint,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setConnection,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { foregroundSessionScopes } from '@/store/session-states'

import { clearSingleFlightSessionResumeState } from '../session/hooks/use-prompt-actions/single-flight-resume'
import { useSubmitPrompt } from '../session/hooks/use-prompt-actions/submit'
import { useRouteResume } from '../session/hooks/use-route-resume'
import { useSessionActions } from '../session/hooks/use-session-actions'
import { useSessionStateCache } from '../session/hooks/use-session-state-cache'

import { createOverviewActions } from './actions'

// Navigation is the only UI shell seam. SDK hydration, actual native resume,
// actual state cache, owner routing, and the submit hook stay production code.
vi.mock('@/app/open-session', () => ({
  openSession: (id: string, navigate: (path: string) => void) => navigate('/' + id)
}))
vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getSession: vi.fn(),
  getLatestSessionMessages: vi.fn()
}))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  openGatewayForAgent: vi.fn(async () => undefined),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

const storedId = '20260905_203342_cf705f'
const runtimeId = 'f0c655bc'
const owner = { connectionId: 'e2e-b', profile: 'analyst', targetProfile: 'analyst', mode: 'remote' as const }

const row = {
  key: JSON.stringify(['e2e-b', 'analyst', storedId]),
  id: storedId,
  resolvedId: storedId,
  runtimeId,
  history: 'known-empty',
  connectionId: 'e2e-b',
  profile: 'analyst',
  sourceLabel: 'B',
  owner: 'analyst',
  title: 'Empty live chat',
  description: '',
  preview: '',
  model: '',
  provider: '',
  platform: 'desktop',
  attention: 'idle',
  canonical: false,
  stale: false,
  lastActive: 1
} satisfies AgentRow

let resumeDone: Promise<unknown> | undefined
let submit: ReturnType<typeof useSubmitPrompt>

const ambientRequest = vi.fn(async () => {
  throw new Error('Ambient dispatch forbidden')
})

function Harness() {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const busyRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    selectedStoredSessionId,
    busyRef,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const routeToken = () => window.location.hash.slice(1)

  const actions = useSessionActions({
    ...cache,
    activeSessionId,
    selectedStoredSessionId,
    busyRef,
    creatingSessionRef: useRef(false),
    navigate: ((path: string) => {
      window.location.hash = path
    }) as never,
    getRouteToken: routeToken,
    getRoutedStoredSessionId: () => storedId,
    requestGateway: ambientRequest
  })

  const request = useStore($sessionResumeRequest)
  useRouteResume({
    ...cache,
    activeSessionId,
    selectedStoredSessionId,
    creatingSessionRef: useRef(false),
    currentView: routeToken() === '/' + storedId ? 'chat' : 'agents',
    freshDraftReady: false,
    gatewayState: 'open',
    locationPathname: routeToken(),
    routedSessionId: routeToken() === '/' + storedId ? storedId : null,
    sessionResumeRequest: request,
    resumeFailedSessionId: null,
    resumeExhaustedSessionId: useStore($resumeExhaustedSessionId),
    resumeSession: (...args) => {
      resumeDone = actions.resumeSession(...args)

      return resumeDone
    },
    startFreshSessionDraft: actions.startFreshSessionDraft
  })
  submit = useSubmitPrompt({
    ...cache,
    busyRef,
    copy: useI18n().t.desktop,
    createBackendSessionForSend: actions.createBackendSessionForSend,
    getRoutedStoredSessionId: () => storedId,
    getRouteToken: routeToken,
    getRuntimeIdForStoredSession: id => cache.runtimeIdByStoredSessionIdRef.current.get(id) ?? null,
    resumeStoredSession: async id => {
      await actions.resumeSession(id, true, owner)
    },
    requestGateway: createSessionRpcDispatcher({ ...cache, ambientRequest }),
    syncAttachmentsForSubmit: async (sessionId, attachments) => ({ sessionId, attachments })
  })

  return null
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
  clearSingleFlightSessionResumeState()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setBusy(false)
  $gatewaySwapTarget.set(null)
  $resumeExhaustedSessionId.set(null)
  resumeDone = undefined
})

async function openEmpty(expectEmptyOverride = false, openingRow: AgentRow = row) {
  vi.useFakeTimers()
  setConnection({ connectionId: 'e2e-a', mode: 'remote' } as never)
  setSessions([])
  setMessages([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  $showAllProfiles.set(true)
  window.location.hash = '/agents'
  vi.mocked(getSession).mockRejectedValue(new Error('404: Session not found'))
  vi.mocked(getLatestSessionMessages).mockRejectedValue(new Error('404: Session not found'))
  vi.mocked(requestGatewayForAgent).mockImplementation(async (connection, profile, method, params) => {
    expect(connection).toBe(owner.connectionId)
    expect(profile).toBe(owner.profile)

    if (method === 'session.resume') {
      expect(params).toMatchObject({ session_id: storedId, profile: 'analyst' })

      // Actual live-session response shape: no eager cwd/model; empty history.
      return {
        session_id: runtimeId,
        stored_session_id: storedId,
        info: { lazy: true },
        messages: [],
        message_count: 0
      } as never
    }

    if (method === 'prompt.submit') {
      return { status: 'streaming' } as never
    }

    return {} as never
  })
  render(<Harness />)

  const actions = createOverviewActions({
    request: vi.fn(),
    open: (id, options) =>
      host.openSession(id, {
        ...options,
        ...(expectEmptyOverride ? { expectHistory: false } : {}),
        hydrationTimeoutMs: 100
      })
  })

  let outcome: 'pending' | 'resolved' | string = 'pending'
  let opening!: Promise<void>
  await act(async () => {
    opening = actions.open(openingRow).then(
      () => {
        outcome = 'resolved'
      },
      error => {
        outcome = error.message
      }
    )
  })
  await act(async () => {
    await resumeDone
  })
  expect(getSession).toHaveBeenCalledWith(storedId, { connectionId: 'e2e-b', profile: 'analyst' })
  expect(getSessionOwnerHint(storedId)).toMatchObject(owner)
  expect($activeSessionId.get()).toBe(runtimeId)
  expect($selectedStoredSessionId.get()).toBe(storedId)
  expect($messages.get()).toEqual([])
  expect(ambientRequest).not.toHaveBeenCalled()

  return { opening, outcome: () => outcome }
}

it('completes native Open for an empty live runtime after metadata 404', async () => {
  const run = await openEmpty()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(101)
    await run.opening
  })
  expect(run.outcome()).toBe('resolved')
  expect($resumeExhaustedSessionId.get()).toBeNull()
})

it('isolates the first-send symptom from the history hydration gate', async () => {
  const run = await openEmpty(false, { ...row, history: 'unknown' })
  expect($gatewaySwapTarget.get()).toBe('analyst')
  let sent: unknown
  await act(async () => {
    sent = await submit('first empty-session turn')
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(101)
    await run.opening
  })
  expect(sent).toBe(true)
  expect(vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'prompt.submit')).toHaveLength(1)
})

it('counterfactual empty-history expectation makes the same native runtime ready without cwd', async () => {
  const run = await openEmpty(true)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(101)
    await run.opening
  })
  expect(run.outcome()).toBe('resolved')
  expect($gatewaySwapTarget.get()).toBeNull()
})

it('retains an explicitly owned foreground runtime before any session event', async () => {
  const run = await openEmpty(true)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(101)
    await run.opening
  })
  expect(foregroundSessionScopes().has(registryBackendScopeKey(owner.connectionId, owner.profile))).toBe(true)
  await act(async () => {
    window.location.hash = '/newer-selection'
    setSelectedStoredSessionId('newer-selection')
  })
  expect($selectedStoredSessionId.get()).toBe('newer-selection')
  expect(foregroundSessionScopes().has(registryBackendScopeKey(owner.connectionId, owner.profile))).toBe(false)
})

it.each(['known-history', 'unknown'] as const)(
  'does not mistake a bound empty runtime for %s hydration',
  async history => {
    const run = await openEmpty(false, { ...row, history })
    expect(run.outcome()).toBe('pending')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(101)
      await run.opening
    })
    expect(run.outcome()).toMatch(/Timed out loading/)
    expect($resumeExhaustedSessionId.get()).toBe(storedId)
  }
)
