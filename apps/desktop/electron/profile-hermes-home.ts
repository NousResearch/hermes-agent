import path from 'node:path'

// Canonical profile HERMES_HOME: default/empty is the install root; a named
// profile lives at <root>/profiles/<name>. Mirrors hermes_cli.profiles
// get_profile_dir / resolve_profile_env and the fs-ipc localPluginsRoot rule.
// Do not invert this (profile-migration.resolveHermesHome maps profilesRoot→root).
export function resolveProfileHermesHome(
  root: string,
  profile?: string | null,
  flavor: 'posix' | 'win32' | 'local' = 'posix'
): string {
  const join = flavor === 'win32' ? path.win32.join : flavor === 'local' ? path.join : path.posix.join

  if (profile && profile !== 'default') {
    return join(root, 'profiles', profile)
  }

  return root
}
