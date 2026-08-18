export function poolTouchKeys(scope: unknown): string[] {
  const key = String(scope ?? '').trim()

  if (!key) {
    return []
  }

  const localPrefix = 'conn:local::'

  if (key.startsWith(localPrefix)) {
    const delegatedProfile = key.slice(localPrefix.length)

    if (delegatedProfile) {
      return [key, delegatedProfile]
    }
  }

  return [key]
}
