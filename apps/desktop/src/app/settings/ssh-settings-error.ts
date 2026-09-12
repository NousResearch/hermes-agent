const MAPPED_SSH_ERRORS = new Set([
  'auth-failed',
  'hermes-not-found',
  'host-key-changed',
  'timeout',
  'unreachable',
  'unsupported-platform',
  'update-required'
])

export function presentSshSettingsError({
  sshError,
  mapped,
  unknown,
  detail
}: {
  sshError: string
  mapped: Record<string, string>
  unknown: string
  detail?: unknown
}): { message: string; detail?: string } {
  const message = mapped[sshError] || unknown

  if (MAPPED_SSH_ERRORS.has(sshError)) {
    return { message }
  }

  const excerpt = typeof detail === 'string' ? detail.trim() : ''

  return excerpt ? { message, detail: excerpt } : { message }
}
