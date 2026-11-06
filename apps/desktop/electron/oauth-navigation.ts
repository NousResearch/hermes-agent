export function isExpectedOauthNavigationAbort(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const code = 'code' in error ? Number((error as { code?: unknown }).code) : Number.NaN
  const message = error instanceof Error ? error.message : String(error)

  return code === -3 || /\bERR_ABORTED\b|\(-3\)/.test(message)
}

// What a scale-to-zero cloud agent answers while its machine resumes, so the same load
// succeeds a moment later. Behind an edge proxy the connection is accepted, then reset, closed
// or left empty. ERR_CONNECTION_REFUSED waits too, on purpose: a host without an edge in front
// refuses until the agent listens. The cost is that a wrong URL waits out the budget.
const WAKING_AGENT_NET_ERROR = /\bERR_(CONNECTION_(CLOSED|RESET|REFUSED)|EMPTY_RESPONSE)\b/

export function isWakingAgentLoadFailure(error: unknown): boolean {
  return WAKING_AGENT_NET_ERROR.test(error instanceof Error ? error.message : String(error))
}
