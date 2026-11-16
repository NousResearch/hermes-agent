import fs from 'node:fs'
import path from 'node:path'

// Port of hermes_constants._PROFILE_IDENTITY_MARKERS: runtime side-effects
// (cron heartbeats, log rotation, caches) create profile dirs carrying none.
const PROFILE_IDENTITY_MARKERS = ['config.yaml', '.env', 'SOUL.md', 'profile.yaml', 'auth.json', 'state.db']
// hermes_constants._DELETED_PROFILES_DIR: `hermes profile delete` tombstones here.
const DELETED_PROFILES_DIR = '.deleted'

// Each check fails on its own, like the os.path predicates behind pathlib on the
// shipped Python 3.14: any OSError (ELOOP, EACCES, ...) reads that one path as
// absent and the remaining checks still run. throwIfNoEntry only covers ENOENT.
const stat = (target: string, follow: boolean): fs.Stats | undefined => {
  try {
    return (follow ? fs.statSync : fs.lstatSync)(target, { throwIfNoEntry: false })
  } catch {
    return undefined
  }
}

/**
 * Whether `<hermesHome>/profiles/<profile>` is a profile the CLI would start.
 * Mirrors hermes_constants.named_profile_is_live exactly: an existing
 * directory with an identity marker (a dangling symlinked marker counts) and
 * no tombstone under `profiles/.deleted/`. A directory-only check would keep a
 * deleted or marker-less profile remembered, and `--profile` would still
 * refuse it on every launch.
 */
export function localProfileIsLive(hermesHome: string, profile: string): boolean {
  const profilesRoot = path.join(hermesHome, 'profiles')
  const home = path.join(profilesRoot, profile)

  if (!stat(home, true)?.isDirectory()) {
    return false
  }

  const hasIdentity = PROFILE_IDENTITY_MARKERS.some(marker => {
    const target = path.join(home, marker)

    return Boolean(stat(target, true)?.isFile() || stat(target, false)?.isSymbolicLink())
  })

  return hasIdentity && !stat(path.join(profilesRoot, DELETED_PROFILES_DIR, profile), true)
}
