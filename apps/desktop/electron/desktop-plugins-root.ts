// The standalone desktop-plugin root (`<HERMES_HOME>/desktop-plugins`) and its
// one-time migration out of the per-profile folders earlier builds used.
//
// A desktop plugin extends THIS APP — panes, palette commands, themes — not an
// agent. Profiles are agents; a plugin that appeared and disappeared with the
// active profile read as "my plugin vanished" every time the user switched.
// Bundled plugins never had that problem (they ship in the app), which hid
// the bug for disk installs.
import fs from 'node:fs'
import path from 'node:path'

export const DESKTOP_PLUGINS_DIR = 'desktop-plugins'

export async function ensureDir(dir: string): Promise<string> {
  try {
    await fs.promises.mkdir(dir, { recursive: true })
  } catch {
    // Best-effort create; return the path regardless so a reveal action can
    // still surface a real openPath error and the scanner can retry later.
  }

  return dir
}

/** Move every `profiles/<name>/desktop-plugins/<id>` folder into the app-level
 *  root. A plugin already present at the root wins (folders are keyed by plugin
 *  id, so a duplicate is the same plugin installed twice); the profile copy is
 *  left in place for the user to delete rather than destroyed. Emptied profile
 *  roots are removed so the migration is a no-op on the next launch. */
export async function migrateProfileScopedDesktopPlugins(hermesHome: string, appRoot: string): Promise<string[]> {
  const profilesDir = path.join(hermesHome, 'profiles')
  const moved: string[] = []

  let profiles: fs.Dirent[]

  try {
    profiles = await fs.promises.readdir(profilesDir, { withFileTypes: true })
  } catch {
    return moved
  }

  for (const profile of profiles) {
    if (!profile.isDirectory()) {
      continue
    }

    const scopedRoot = path.join(profilesDir, profile.name, DESKTOP_PLUGINS_DIR)

    let entries: fs.Dirent[]

    try {
      entries = await fs.promises.readdir(scopedRoot, { withFileTypes: true })
    } catch {
      continue
    }

    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue
      }

      const from = path.join(scopedRoot, entry.name)
      const to = path.join(appRoot, entry.name)

      if (fs.existsSync(to)) {
        continue
      }

      try {
        await fs.promises.rename(from, to)
        moved.push(to)
      } catch {
        // Cross-device or permission failure: leave the folder; the user can
        // still reach it through the profile directory.
      }
    }

    try {
      if ((await fs.promises.readdir(scopedRoot)).length === 0) {
        await fs.promises.rmdir(scopedRoot)
      }
    } catch {
      // Not empty or already gone — either is fine.
    }
  }

  return moved
}
