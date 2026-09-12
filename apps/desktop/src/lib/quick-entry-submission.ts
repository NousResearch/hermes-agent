export type QuickEntryAcceptedPromptStatus = 'streaming' | 'queued' | 'steered' | 'redirected'

export interface QuickEntryRequestIdentity {
  /** The exact prompt text submitted through Quick Entry. */
  readonly prompt: string
  /** The selected Quick Entry target: current, new, or a stored session id. */
  readonly target: string
  /** The durable session id the caller intends to address. */
  readonly storedSessionId: string | null
}

export interface CapturedQuickEntryRequest {
  readonly correlationId: string
  readonly identity: QuickEntryRequestIdentity
}

export interface QuickEntryBackendPromptResult {
  readonly correlationId: string
  /** The durable session id accepted by the backend. */
  readonly sessionId: string
  readonly runtimeSessionId: string
  readonly status: QuickEntryAcceptedPromptStatus
}

export interface QuickEntryBackendResultEvidence {
  readonly observed: true
  readonly status: QuickEntryAcceptedPromptStatus
}

export interface QuickEntryAcceptedIdentity {
  readonly runtimeSessionId: string
  readonly storedSessionId: string
}

export type QuickEntrySubmitOutcome =
  | {
      readonly acceptedIdentity: QuickEntryAcceptedIdentity
      readonly correlationId: string
      readonly evidence: QuickEntryBackendResultEvidence
      readonly ok: true
      readonly requestedIdentity: QuickEntryRequestIdentity
      readonly status: 'accepted'
    }
  | {
      readonly ok: false
      readonly reason: 'no-prompt-dispatched'
      readonly retryable: true
      readonly status: 'rejected'
    }
  | {
      readonly ok: false
      readonly reason:
        | 'correlation-mismatch'
        | 'invalid-backend-result'
        | 'invalid-request'
        | 'stored-identity-mismatch'
        | 'unsupported-backend-status'
      readonly retryable: false
      readonly status: 'unknown'
    }

export interface QuickEntryDispatchContext {
  /**
   * Only a caller that can prove its failure happened before prompt dispatch may
   * set this to false. Every post-dispatch failure is unknown because retry can
   * duplicate the original prompt.
   */
  readonly promptDispatched: boolean
}

type UnknownRecord = Record<string, unknown>

const ACCEPTED_PROMPT_STATUSES: ReadonlySet<string> = new Set(['streaming', 'queued', 'steered', 'redirected'])

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

/**
 * Capture the mutable Quick Entry inputs before any await can interleave with a
 * session switch. The returned request and nested identity are both frozen.
 */
export function captureQuickEntryRequest(input: {
  correlationId: string
  identity: QuickEntryRequestIdentity
}): CapturedQuickEntryRequest {
  const identity: QuickEntryRequestIdentity = { ...input.identity }
  const request: CapturedQuickEntryRequest = {
    correlationId: input.correlationId,
    identity
  }

  Object.freeze(identity)
  Object.freeze(request)

  return request
}

function invalidRequestOutcome(dispatch: QuickEntryDispatchContext): QuickEntrySubmitOutcome {
  return dispatch.promptDispatched
    ? {
        ok: false,
        reason: 'invalid-request',
        retryable: false,
        status: 'unknown'
      }
    : {
        ok: false,
        reason: 'no-prompt-dispatched',
        retryable: true,
        status: 'rejected'
      }
}

function unknownOutcome(
  reason: Extract<QuickEntrySubmitOutcome, { status: 'unknown' }>['reason']
): QuickEntrySubmitOutcome {
  return {
    ok: false,
    reason,
    retryable: false,
    status: 'unknown'
  }
}

/**
 * Turn a backend response into an acceptance receipt. A supported status alone
 * is not enough: the result must also carry this request's correlation and the
 * exact durable/runtime session identities.
 */
export function validateQuickEntryAcceptance(
  request: unknown,
  backendResult: unknown,
  dispatch: QuickEntryDispatchContext
): QuickEntrySubmitOutcome {
  const requestRecord = isRecord(request) ? request : {}
  const identity = requestRecord.identity
  const correlationId = requestRecord.correlationId

  if (
    !isRecord(identity) ||
    !nonEmptyString(identity.prompt) ||
    !nonEmptyString(identity.target) ||
    !nonEmptyString(identity.storedSessionId) ||
    !nonEmptyString(correlationId)
  ) {
    return invalidRequestOutcome(dispatch)
  }

  if (dispatch.promptDispatched === false) {
    return {
      ok: false,
      reason: 'no-prompt-dispatched',
      retryable: true,
      status: 'rejected'
    }
  }

  const result = isRecord(backendResult) && Object.keys(backendResult).length > 0 ? backendResult : null
  if (
    result === null ||
    !nonEmptyString(result.correlationId) ||
    !nonEmptyString(result.sessionId) ||
    !nonEmptyString(result.runtimeSessionId) ||
    !nonEmptyString(result.status)
  ) {
    return unknownOutcome('invalid-backend-result')
  }

  if (result.correlationId !== correlationId) {
    return unknownOutcome('correlation-mismatch')
  }

  if (result.sessionId !== identity.storedSessionId) {
    return unknownOutcome('stored-identity-mismatch')
  }

  if (!ACCEPTED_PROMPT_STATUSES.has(result.status)) {
    return unknownOutcome('unsupported-backend-status')
  }

  const status = result.status as QuickEntryAcceptedPromptStatus

  return {
    acceptedIdentity: {
      runtimeSessionId: result.runtimeSessionId,
      storedSessionId: result.sessionId
    },
    correlationId,
    evidence: {
      observed: true,
      status
    },
    ok: true,
    requestedIdentity: {
      prompt: identity.prompt,
      storedSessionId: identity.storedSessionId,
      target: identity.target
    },
    status: 'accepted'
  }
}
