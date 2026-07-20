import path from 'node:path'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

/** Resolve the trusted desktop-machine plugin home without consulting a remote gateway. */
export function desktopPluginHome(root: string, selectedProfile: null | string, readStickyProfile?: () => string): string {
  let profile = selectedProfile

  if (!profile && readStickyProfile) {
    try {
      profile = readStickyProfile().trim()
    } catch {
      profile = null
    }
  }

  if (!profile || profile === 'default' || !PROFILE_NAME_RE.test(profile)) {
    return root
  }

  return path.join(root, 'profiles', profile)
}
