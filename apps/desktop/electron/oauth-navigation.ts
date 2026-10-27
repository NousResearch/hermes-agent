export function isExpectedOauthNavigationAbort(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const code = 'code' in error ? Number((error as { code?: unknown }).code) : Number.NaN
  const message = error instanceof Error ? error.message : String(error)

  return code === -3 || /\bERR_ABORTED\b|\(-3\)/.test(message)
}
