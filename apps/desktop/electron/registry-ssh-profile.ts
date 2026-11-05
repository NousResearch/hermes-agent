export function registrySshRemoteProfile(remoteProfile: string | null | undefined, profile: string | null | undefined): string {
  const profileKey = String(profile ?? '').trim() || 'default'

  // remoteProfile is the connection's launch default, not a rewrite of a
  // concrete profile selected from that host's roster or stored-session row.
  return profileKey === 'default' ? remoteProfile || '' : profileKey
}
