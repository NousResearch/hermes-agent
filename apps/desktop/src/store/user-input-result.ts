export type UserInputAcknowledgementKind =
  | 'accepted'
  | 'invalid'
  | 'malformed'
  | 'not_found'
  | 'terminal'
  | 'transport'

export interface UserInputAcknowledgement {
  accepted: boolean
  clear: boolean
  delivery?: string
  kind: UserInputAcknowledgementKind
  message?: string
  recorded?: boolean
  retryable: boolean
  status: string
  terminal?: boolean
}

const knownDeliveries = new Set(['steered', 'redirected', 'queued', 'deferred'])
const terminalStatuses = new Set(['answered', 'expired', 'cancelled'])

const notFoundCodes = new Set([
  'not_found',
  'request_not_found',
  'user_input_not_found',
  'user_input_request_not_found'
])

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

const stringValue = (value: unknown): string => typeof value === 'string' ? value : ''

const normalized = (value: unknown): string => stringValue(value).trim().toLowerCase().replace(/-/g, '_')

const errorRecord = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object') {return {}}

  return value as Record<string, unknown>
}

const resultMessage = (value: unknown): string => {
  const source = record(value)
  const error = record(source.error)
  const message = error.message || source.detail || source.message || source.error

  return typeof message === 'string' ? message.replace(/\s+/g, ' ').trim().slice(0, 500) : ''
}

export function classifyUserInputResult(
  value: unknown,
  httpStatus = 0,
  thrownError: unknown = null
): UserInputAcknowledgement {
  const source = record(value)
  const thrown = errorRecord(thrownError)
  const error = record(source.error)
  const status = normalized(source.status || source.data && record(source.data).status || source.state || error.code || thrown.code)
  const code = normalized(source.code || source.error_code || source.errorCode || error.code || thrown.code)
  const explicitNotFound = status === 'not_found' || notFoundCodes.has(code)
  const accepted = source.accepted === true
  const rejected = source.accepted === false
  const delivery = normalized(source.delivery)
  const message = resultMessage(value) || resultMessage(thrownError)

  if (explicitNotFound) {
    return {
      accepted: false,
      clear: true,
      kind: 'not_found',
      ...(message ? { message } : {}),
      retryable: false,
      status: 'not_found',
      terminal: true
    }
  }

  if (status === 'invalid' || code === 'invalid') {
    return {
      accepted: false,
      clear: false,
      kind: 'invalid',
      message: message || 'Hermes rejected these answers. Check the fields and try again.',
      retryable: true,
      status: 'invalid'
    }
  }

  const terminalStatus = status === 'already_answered' ? 'answered' : status

  if (rejected && terminalStatuses.has(terminalStatus)) {
    return {
      accepted: false,
      clear: true,
      kind: 'terminal',
      retryable: false,
      status: terminalStatus,
      terminal: true
    }
  }

  if (accepted && terminalStatus === 'answered' && (!delivery || knownDeliveries.has(delivery))) {
    return {
      accepted: true,
      clear: true,
      ...(delivery ? { delivery } : {}),
      kind: 'accepted',
      recorded: true,
      retryable: false,
      status: 'answered'
    }
  }

  const numericStatus = Number(httpStatus)
  const httpFailure = Number.isInteger(numericStatus) && numericStatus >= 400

  return {
    accepted: false,
    clear: false,
    kind: thrownError || httpFailure ? 'transport' : 'malformed',
    message: message || (httpFailure ? `Hermes input answer failed (${numericStatus}).` : 'Hermes returned an unrecognized user-input acknowledgement.'),
    retryable: true,
    status: thrownError || httpFailure ? 'transport' : 'malformed'
  }
}

export function createUserInputSubmitLock() {
  const inFlight = new Set<string>()
  const key = (sessionId: string, requestId: string) => `${sessionId}:${requestId}`

  return {
    acquire(sessionId: string, requestId: string): boolean {
      const lockKey = key(sessionId, requestId)

      if (inFlight.has(lockKey)) {return false}
      inFlight.add(lockKey)

      return true
    },
    release(sessionId: string, requestId: string): void {
      inFlight.delete(key(sessionId, requestId))
    }
  }
}
