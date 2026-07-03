export interface SetPrimaryDesktopProfileDeps {
  primaryProfileKey: () => string
  writeActiveDesktopProfile: (name: unknown) => null | string
  teardownPrimaryBackendAndWait: () => Promise<void>
  reloadMainWindow: () => void
}

export async function setPrimaryDesktopProfile(
  name: unknown,
  deps: SetPrimaryDesktopProfileDeps
): Promise<{ profile: null | string }> {
  const previous = deps.primaryProfileKey()
  const next = deps.writeActiveDesktopProfile(name)
  const nextKey = next || 'default'

  if (nextKey !== previous) {
    await deps.teardownPrimaryBackendAndWait()
    deps.reloadMainWindow()
  }

  return { profile: next }
}
