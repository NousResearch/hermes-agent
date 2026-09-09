const FULL_GIT_SHA = /^[0-9a-f]{40}$/i

export function isValidInstallCommit(value: unknown): value is string {
  return typeof value === 'string' && FULL_GIT_SHA.test(value) && !/^0{40}$/.test(value)
}
