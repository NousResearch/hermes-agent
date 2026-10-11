import { type Translations } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'

import type { ClientSessionState } from '../../../types'

import { inlineErrorMessage, isProviderSetupError, isSessionBusyError, isSessionNotOwnedError } from './utils'

export interface HandleSubmitPipelineErrorParams {
  copy: Translations['desktop']
  err: unknown
  fromQueue: boolean
  releaseBusy: () => void
  sessionId: string
  storedSessionId: null | string
  targetIsCurrentView: boolean
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/**
 * Terminal error handling for the submit pipeline (the catch arm of the
 * attach → prompt.submit transaction), extracted from useSubmitPrompt so the
 * hook stays under its complexity cap. Always resolves `false`; the optimistic
 * user row was already bound to a real error or dropped by the caller's
 * earlier abort paths.
 */
export function handleSubmitPipelineError({
  copy,
  err,
  fromQueue,
  releaseBusy,
  sessionId,
  storedSessionId,
  targetIsCurrentView,
  updateSessionState
}: HandleSubmitPipelineErrorParams): false {
  releaseBusy()

  // A queued drain that raced a not-yet-settled turn gets a transient
  // "session busy" (4009). Don't surface an error bubble/toast — the entry
  // stays queued and the composer's bounded auto-drain retries when idle.
  if (fromQueue && isSessionBusyError(err)) {
    return false
  }

  const message = inlineErrorMessage(err, copy.promptFailed)
  const occurredAt = Date.now() / 1000
  // Another surface owns the session (#106217): a deterministic gateway
  // refusal, so the error card drops Retry and offers a new session.
  const notOwned = isSessionNotOwnedError(err)

  updateSessionState(
    sessionId,
    state => ({
      ...state,
      messages: [
        ...state.messages,
        {
          id: `assistant-error-${Date.now()}`,
          role: 'assistant',
          parts: [],
          error: message || copy.promptFailed,
          ...(notOwned && { errorSurface: { layer: 'gateway', code: 'SESSION_NOT_OWNED', retryable: false } }),
          branchGroupId: state.pendingBranchGroup ?? undefined,
          completedAt: occurredAt,
          timestamp: occurredAt
        }
      ],
      busy: false,
      awaitingResponse: false,
      pendingBranchGroup: null,
      sawAssistantPayload: true,
      // The failed submit's clock seed dies with the turn it never got.
      turnStartedAt: null
    }),
    storedSessionId
  )

  if (targetIsCurrentView && isProviderSetupError(err)) {
    requestDesktopOnboarding(copy.providerCredentialRequired)

    return false
  }

  if (targetIsCurrentView) {
    notifyError(err, copy.promptFailed)
  }

  return false
}
