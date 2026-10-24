import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  markComposerEffortManual,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from '@/store/session'

import { handleSessionInfoEvent } from './session-info'
import type { GatewayEventContext } from './types'

const EFFORT_KEY = 'hermes.desktop.composer.reasoning-effort'
const TIER_KEY = 'hermes.desktop.composer.service-tier'
const FAST_KEY = 'hermes.desktop.composer.fast'

// The composer's reasoning-effort (and tier/fast) picks are sticky UI state
// persisted in localStorage and shipped as explicit per-session overrides on
// session.create. session.info reports a session's BACKEND-RESOLVED runtime
// values, which routinely differ from the pick — an older resumed session, a
// per-model reasoning default, a level the create silently dropped (#87036).
// The heartbeat must paint the live values into the session state (the pills
// read the slice) without ever overwriting the persisted pick.
function foregroundInfoEvent(payload: Record<string, unknown>): GatewayEventContext {
  return {
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: 'runtime-1' },
      hydrateFromStoredSession: vi.fn(),
      lastCwdInfoSessionRef: { current: null },
      queryClient: { invalidateQueries: vi.fn() },
      refreshHermesConfig: vi.fn(),
      scheduleSessionsRefresh: vi.fn(),
      sessionInterrupted: () => false,
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    },
    event: { profile: 'default', session_id: 'runtime-1', type: 'session.info' },
    explicitSid: 'runtime-1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: Date.now() / 1000,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'runtime-1'
  } as unknown as GatewayEventContext
}

describe('handleSessionInfoEvent keeps the composer effort pick (#87036)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setCurrentReasoningEffort('max')
    markComposerEffortManual()
    setCurrentServiceTier('priority')
    setCurrentFastMode(false)
    setCurrentModel('anthropic/claude-opus-5')
    setCurrentProvider('anthropic')
    setCurrentModelSource('manual')
  })

  afterEach(() => {
    window.localStorage.clear()
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  it('paints a live session effort into the state slice without touching the persisted pick', () => {
    const stored = createClientSessionState('stored-1')
    let patched: ClientSessionState | undefined

    const ctx = foregroundInfoEvent({
      fast: true,
      model: 'anthropic/claude-opus-5',
      provider: 'anthropic',
      reasoning_effort: 'ultra',
      service_tier: 'flex',
      stored_session_id: 'stored-1'
    })

    ctx.deps.updateSessionState = vi.fn(
      (_sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
        patched = updater(stored)

        return patched
      }
    )

    handleSessionInfoEvent(ctx)

    // The pick survives — atom AND localStorage keep what the user chose.
    expect($currentReasoningEffort.get()).toBe('max')
    expect(window.localStorage.getItem(EFFORT_KEY)).toBe('max')
    // The live session still reports its own effort through the state slice
    // (what the pill renders while the session is open).
    expect(patched?.reasoningEffort).toBe('ultra')
    expect(patched?.serviceTier).toBe('flex')
    expect(patched?.fast).toBe(true)
  })

  it('does not let a foreground session.info overwrite the persisted tier/fast picks', () => {
    handleSessionInfoEvent(foregroundInfoEvent({ fast: true, reasoning_effort: 'ultra', service_tier: 'flex' }))

    expect($currentServiceTier.get()).toBe('priority')
    expect(window.localStorage.getItem(TIER_KEY)).toBe('priority')
    expect($currentFastMode.get()).toBe(false)
    expect(window.localStorage.getItem(FAST_KEY)).toBe('false')
  })

  it('keeps the manual model pick untouched by the same heartbeat (the #102793 guard)', () => {
    handleSessionInfoEvent(
      foregroundInfoEvent({
        model: 'openai/gpt-5.6-sol',
        provider: 'openai-codex',
        reasoning_effort: 'ultra'
      })
    )

    expect($currentModel.get()).toBe('anthropic/claude-opus-5')
    expect($currentProvider.get()).toBe('anthropic')
    expect(window.localStorage.getItem('hermes.desktop.composer.model')).toBe('anthropic/claude-opus-5')
  })
})
