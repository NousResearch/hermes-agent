import { atom } from 'nanostores'

import { getGlobalModelInfo } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { segmentTranscriptDirectives } from '@/lib/transcript-directives'
import { type FreeTierRequester } from '@/store/free-tier'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { guidedOnboardingActive } from '@/store/onboarding-gate'
import { onboardingSurfaceActive } from '@/store/onboarding-presence'
import { $activeGatewayProfile, $newChatProfile, resolveNewChatOwnerRoute } from '@/store/profile'
import { $currentModel, $currentProvider, setCurrentModel, setCurrentProvider } from '@/store/session'
import { $sessionStates, knownOwnerForSession } from '@/store/session-states'
import type { FreeTierStatus } from '@/types/hermes'

export interface ContinuationOwner {
  connectionId: string | null
  profile: string
}

export interface ContinuationTarget {
  owner: ContinuationOwner
  sessionId: string | null
  model?: string
  provider?: string
}

interface ContinuationVerdict {
  target: ContinuationTarget
  status: FreeTierStatus
}

interface ContinuationInput {
  text: string
  hidden?: boolean
}

// A route verdict is not an identity counter: one capped identity can still
// supply connectors beside a local/BYO session. Never gate from $freeTierStatus.
export const $freeTierContinuation = atom<Record<string, ContinuationVerdict>>({})
export const $continuationRefresh = atom(0)
const generations = new Map<string, number>()

export function continuationKey(target: ContinuationTarget): string {
  return JSON.stringify([target.owner.connectionId, target.owner.profile, target.sessionId, target.model, target.provider])
}

export function continuationTarget(sessionId: string | null): ContinuationTarget | null {
  if (sessionId) {
    const owner = knownOwnerForSession(sessionId)

    if (!owner) {
      return null
    }

    return {
      owner: typeof owner === 'string'
        ? { connectionId: null, profile: owner }
        : { connectionId: owner.connectionId ?? null, profile: owner.targetProfile || owner.profile },
      sessionId
    }
  }

  const route = resolveNewChatOwnerRoute()

  return {
    owner: {
      connectionId: route?.connectionId ?? null,
      profile: route?.profile || $newChatProfile.get() || $activeGatewayProfile.get()
    },
    sessionId: null,
    model: $currentModel.get(),
    provider: $currentProvider.get()
  }
}

export function continuationSuppressed(target: ContinuationTarget | null = null): boolean {
  const status = target ? $freeTierContinuation.get()[continuationKey(target)]?.status : undefined

  return (guidedOnboardingActive() || onboardingSurfaceActive())
    && !(status?.continuation_required && status.onboarding_complete)
}

export function continuationRequired(target: ContinuationTarget | null): boolean {
  return Boolean(target && $freeTierContinuation.get()[continuationKey(target)]?.status.continuation_required)
}

export function recordContinuationRefusal(target: ContinuationTarget, status: FreeTierStatus): void {
  const key = continuationKey(target)
  generations.set(key, (generations.get(key) ?? 0) + 1)
  $freeTierContinuation.set({ ...$freeTierContinuation.get(), [key]: {
    target, status: { ...status, continuation_required: true }
  } })
}

export function requestContinuationRefresh(): void {
  $continuationRefresh.set($continuationRefresh.get() + 1)
}

export function continuationRequester(target: ContinuationTarget): FreeTierRequester {
  const { connectionId, profile } = target.owner

  return (method, params) => requestGatewayForAgent(connectionId, profile, method, params)
}

export async function refreshContinuation(
  target: ContinuationTarget,
  request: FreeTierRequester = continuationRequester(target)
): Promise<boolean> {
  const key = continuationKey(target)
  const generation = (generations.get(key) ?? 0) + 1
  generations.set(key, generation)

  try {
    const status = await request<FreeTierStatus>('free_tier.status', target.sessionId
      ? { session_id: target.sessionId }
      : { model: target.model, provider: target.provider })

    // Older backends don't implement the route verdict. A missing value or
    // transport failure is never permission to clear a previously known cap.
    if (generations.get(key) === generation && typeof status?.continuation_required === 'boolean') {
      const current = $freeTierContinuation.get()
      const previous = current[key]?.status

      if (
        previous?.continuation_required !== status.continuation_required
        || previous?.tool_calls_used !== status.tool_calls_used
        || previous?.onboarding_complete !== status.onboarding_complete
      ) {
        $freeTierContinuation.set({ ...current, [key]: { target, status } })
      }
    }
  } catch {
    // Preserve only THIS owner's last answer through a reconnect.
  }

  return continuationRequired(target)
}

/** Only the actual first-task response starts the clarification grace. Setup
 * notes and the explicit "figure it out" fallback are not task choices. */
function choosingFirstTask(sessionId: string | null, input?: ContinuationInput): boolean {
  if (!sessionId || !input || input.hidden || !input.text.trim() || input.text.trim() === "Let's figure it out together") {
    return false
  }

  const messages = $sessionStates.get()[sessionId]?.messages ?? []
  const latest = messages.findLast(message => !message.hidden && (message.role === 'assistant' || message.role === 'user'))

  if (!latest || latest.role !== 'assistant') {
    return false
  }

  return chatMessageText(latest).split(/\n\s*\n/).some(paragraph =>
    segmentTranscriptDirectives(paragraph)?.some(segment => segment.kind === 'directive'
      && segment.directive.name === 'onboarding' && segment.directive.attrs.step === 'first'))
}

/** Runs before optimistic insertion or attachment/draft mutation, including
 * queue drains. The gateway still enforces the same rule for races/other clients. */
export async function blockContinuationSend(sessionId: string | null, input?: ContinuationInput): Promise<boolean> {
  const target = continuationTarget(sessionId)

  if (!target) {
    return false // Unresolved owners are handled by the submit router.
  }

  if (target.owner.profile === 'hermes-setup' && choosingFirstTask(sessionId, input)) {
    try {
      const result = await continuationRequester(target)<{ chosen: boolean }>('free_tier.choose_onboarding_task', { session_id: sessionId })

      if (result?.chosen !== true) {
        throw new Error('Could not confirm your task choice. Please retry.')
      }
    } catch (error) {
      if (!isMissingRpcMethod(error)) {
        notifyError(error, 'Could not save your task choice. Your draft is kept; please retry.')

        return true
      }
    }
  }

  return refreshContinuation(target)
}

/** Successful setup changes a profile default, not an existing session's
 * override. Re-home only the initiating chat, then ask the backend again. */
export async function applyContinuationProvider(target: ContinuationTarget): Promise<void> {
  const info = await getGlobalModelInfo(target.owner)
  const request = continuationRequester(target)

  if (target.sessionId && info.model && info.provider) {
    await request('config.set', {
      session_id: target.sessionId,
      key: 'model',
      value: `${info.model} --provider ${info.provider} --session`
    })
  }

  if (!target.sessionId && continuationKey(continuationTarget(null)!) === continuationKey(target)) {
    setCurrentModel(info.model)
    setCurrentProvider(info.provider)
  }

  await refreshContinuation(target, request)
  requestContinuationRefresh()
}
