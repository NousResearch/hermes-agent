export function isExpectedOauthNavigationAbort(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const code = 'code' in error ? Number((error as { code?: unknown }).code) : Number.NaN
  const message = error instanceof Error ? error.message : String(error)

  return code === -3 || /\bERR_ABORTED\b|\(-3\)/.test(message)
}

// What a scale-to-zero cloud agent answers while its machine resumes: the edge
// accepts the connection but nothing is listening yet, so the same load succeeds
// a moment later.
const WAKING_AGENT_NET_ERRORS = [
  'ERR_CONNECTION_CLOSED',
  'ERR_CONNECTION_RESET',
  'ERR_CONNECTION_REFUSED',
  'ERR_EMPTY_RESPONSE'
]

export function isWakingAgentLoadFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return WAKING_AGENT_NET_ERRORS.some(name => new RegExp(`\\b${name}\\b`).test(message))
}
