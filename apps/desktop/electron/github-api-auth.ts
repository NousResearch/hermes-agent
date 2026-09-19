// Credentials for the self-update check's api.github.com calls.
//
// The passive check asks the REST API for the branch tip SHA. Unauthenticated
// that budget is 60 requests/hour keyed on the *client IP*, so a shared exit —
// office NAT, VPN, a proxy node many users sit behind — exhausts the bucket for
// everyone on it and the check reports a rate limit that reads as "Hermes can't
// reach the update server". Authenticating moves the caller onto the token's
// 5,000/hour budget.
//
// Only the env rung of the Python client's ladder
// (tools/skills_hub_github.py::GitHubAuth) is wired: GITHUB_TOKEN, then
// GH_TOKEN, read from the process env per request and never stored. Whether a
// GUI app may also spend the user's `gh` CLI login on a passive background
// check is a product call, kept out of this module.

/** Env vars consulted, in precedence order. */
export const GITHUB_TOKEN_ENV_VARS = ['GITHUB_TOKEN', 'GH_TOKEN'] as const

/** First non-blank env token, trimmed. A blank value falls through to the next. */
export function githubTokenFromEnv(env: Record<string, string | undefined> = {}): string | null {
  for (const name of GITHUB_TOKEN_ENV_VARS) {
    const value = env[name]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return null
}

/**
 * Headers for api.github.com. The `token` scheme (not `Bearer`) matches the
 * Python client. Anonymous callers get the base headers untouched: an empty
 * `Authorization` header is a 401, so it must be absent, not blank.
 */
export function githubApiHeaders(base: Record<string, string>, token?: string | null): Record<string, string> {
  const headers = { ...base }

  if (token) {
    headers.Authorization = `token ${token}`
  }

  return headers
}

/**
 * True when api.github.com rejected the env token itself (HTTP 401 on an
 * authenticated call). A stale or revoked GITHUB_TOKEN must not fail the
 * update check closed — the caller retries anonymously, which is exactly what
 * worked before the token was wired in. Anonymous 401s and every other status
 * are not the token's fault and are surfaced as-is.
 */
export function envTokenRejected(error: { statusCode?: number; authenticated?: boolean } | null | undefined): boolean {
  return error?.statusCode === 401 && error?.authenticated === true
}
