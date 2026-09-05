/** True when a backend error is the post-update stale-module 503 (#97046). */
export function isCodeSkewRestartRequired(error: unknown): boolean {
  return /Restart required:/i.test(errorText(error))
}

function errorText(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return typeof error === 'string' ? error : String(error ?? '')
}
