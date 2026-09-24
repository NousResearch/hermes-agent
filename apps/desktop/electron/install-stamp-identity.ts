const FULL_GIT_SHA = /^[0-9a-f]{40}$/i

export function isValidInstallCommit(value: unknown): value is string {
  return typeof value === 'string' && FULL_GIT_SHA.test(value) && !/^0{40}$/.test(value)
}

/** The installed renderer is immutable while the staging checkout may move.
 * Development builds and missing/fallback stamps still follow their checkout. */
export function resolveRunningClientSha({
  checkoutSha,
  installStamp,
  isPackaged
}: {
  checkoutSha: string
  installStamp: { commit?: unknown } | null
  isPackaged: boolean
}): string {
  return isPackaged && isValidInstallCommit(installStamp?.commit) ? installStamp.commit.toLowerCase() : checkoutSha
}
