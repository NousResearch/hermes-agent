/** True when a backend error is the post-update stale-module 503 (#97046). */
export function isCodeSkewRestartRequired(error: unknown): boolean {
  return /Restart required:/i.test(errorText(error))
}

/** OS pid of the process that served a code-skew 503, if the backend named it (#101561). */
export function servingPidFromCodeSkewError(error: unknown): number | null {
  const match = /\(pid=(\d+)\)/.exec(errorText(error))

  if (!match) {
    return null
  }

  const pid = Number(match[1])

  return Number.isInteger(pid) && pid > 1 ? pid : null
}

function errorText(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return typeof error === 'string' ? error : String(error ?? '')
}
