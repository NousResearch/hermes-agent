import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'

import type { PreparedSubmission } from './prepared-submissions'
import {
  type GatewayRequest,
  isAdmittedSubmissionConflict,
  type SessionRecoveryDeps,
  type SubmitTextOptions,
  withSessionBusyRetry,
  withSessionNotFoundResume
} from './utils'

/** Confirmed text requires a canonical, identity-bound acknowledgement. */
export function assertConfirmedSubmitReceipt(
  options: SubmitTextOptions | undefined,
  result: { status?: string; admission_id?: string; submission_id?: string; session_id?: string } | undefined,
  submissionId: string,
  sessionId: string
): void {
  if (!options?.confirmedExternal) {
    return
  }

  if (!result || !['started', 'queued', 'terminal'].includes(result.status ?? '') ||
      typeof result.admission_id !== 'string' || !result.admission_id ||
      (result.submission_id ?? result.admission_id) !== submissionId ||
      (result.session_id !== undefined && result.session_id !== sessionId)) {
    throw new Error('Native submission outcome unknown')
  }
}

/** A refusal issued before admitting anything: only ordinary sends may downgrade. */
function isPreAdmissionRefusal(error: unknown): boolean {
  if (!error || typeof error !== 'object' || !('code' in error)) {
    return false
  }

  const { code, message } = error as { code?: unknown; message?: unknown }

  return code === 4094 || (code === 4000 && typeof message === 'string' && /submission_id/.test(message))
}

/** Confirmed sends get exactly one wire attempt, without recovery or legacy
 * downgrade. Receipt identity/status validation belongs to the native pipeline. */
export async function submitPromptTransport<T>({
  options,
  prepared,
  persistLegacyAttempt,
  onLegacyAccepted,
  recovery,
  requestGateway,
  sessionId,
  storedSessionId
}: {
  options?: SubmitTextOptions
  prepared: PreparedSubmission
  persistLegacyAttempt: () => Promise<void>
  onLegacyAccepted: () => void
  recovery: SessionRecoveryDeps
  requestGateway: GatewayRequest
  sessionId: string
  storedSessionId: null | string
}): Promise<{ result: T; sessionId: string }> {
  const submitOnce = (liveId: string, params = prepared.params) =>
    requestGateway<T>('prompt.submit', { ...params, session_id: liveId }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)

  if (options?.confirmedExternal) {
    return { result: await submitOnce(sessionId), sessionId }
  }

  // Keep main's normal-send busy/recovery policy and pre-admission legacy fallback.
  return withSessionNotFoundResume(
    sessionId,
    storedSessionId,
    liveId => withSessionBusyRetry(async () => {
      if (prepared.legacyAttempted) {
        throw new Error('Legacy submission acknowledgement is unknown; automatic retry is unsafe')
      }

      try {
        return await submitOnce(liveId)
      } catch (error) {
        // Ordinary retries of an already admitted identity settle as the
        // original send. Confirmed sends above never convert a refusal to an ACK.
        if (isAdmittedSubmissionConflict(error)) {
          return { status: 'terminal', submission_id: prepared.id } as T
        }

        if (!isPreAdmissionRefusal(error) || prepared.legacyAttempted) {
          throw error
        }

        prepared.legacyAttempted = true
        await persistLegacyAttempt()
        const { submission_id: _id, ...legacyParams } = prepared.params
        const result = await submitOnce(liveId, legacyParams)
        onLegacyAccepted()

        return result
      }
    }),
    recovery,
    { alsoTimeout: true }
  )
}
