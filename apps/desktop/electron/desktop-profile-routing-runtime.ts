import fs from 'node:fs'
import path from 'node:path'

import { connectionScopeKey, profileRemoteOverride, profileSshOverride } from './connection-config'
import { createDesktopProfilePreferences, type DesktopProfileRoute } from './desktop-profile'
import { migrateActiveProfileIfMissing as migrateActiveProfileIfMissingPure } from './profile-migration'

export interface DesktopProfileRoutingRuntimeDeps {
  configPath: string
  hermesHome: string
  profileNameRe: RegExp
  BrowserWindow: any
  readDesktopConnectionConfig: any
  readDesktopConnectionsRegistry: any
  primaryProfileKey: any
  globalRemoteActive: any
  primaryBackendIsRemote: any
  getIsolatedBackend: any
  writeFileAtomic: any
}

export function createDesktopProfileRoutingRuntime(deps: DesktopProfileRoutingRuntimeDeps) {
  const DESKTOP_PROFILE_CONFIG_PATH = deps.configPath
  const HERMES_HOME = deps.hermesHome
  const PROFILE_NAME_RE = deps.profileNameRe

  const {
    BrowserWindow,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    primaryProfileKey,
    globalRemoteActive,
    primaryBackendIsRemote,
    getIsolatedBackend,
    writeFileAtomic
  } = deps

  // Last-used profile and explicit app-wide default share the existing desktop
  // preference file, but only the explicit action changes the default route.
  const desktopProfilePreferences = createDesktopProfilePreferences(DESKTOP_PROFILE_CONFIG_PATH, {
    validateRoute: validateDesktopProfileRoute,
    onDefaultChanged: route => {
      for (const win of BrowserWindow.getAllWindows()) {
        if (!win.webContents.isDestroyed()) {
          win.webContents.send('hermes:profile:default:changed', route)
        }
      }
    }
  })

  function validateDesktopProfileRoute(route: DesktopProfileRoute) {
    if (
      route.connectionId &&
      !readDesktopConnectionsRegistry().connections.some((source: { id: string }) => source.id === route.connectionId)
    ) {
      throw new Error(`No connection with id "${route.connectionId}".`)
    }
  }

  function readActiveDesktopProfile() {
    return desktopProfilePreferences.readActive()
  }

  function writeActiveDesktopProfile(name) {
    return desktopProfilePreferences.remember(name)
  }

  // True when the given pid belongs to a running process whose command line
  // contains "hermes", avoiding false positives from stale gateway.pid files
  // whose PID was recycled by the OS to an unrelated process.
  function isHermesProcess(pid) {
    try {
      process.kill(pid, 0) // signal 0 = existence check, no signal sent
    } catch {
      return false
    }

    // On macOS / Linux, check the command line to avoid PID recycling false positives.
    try {
      const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8')

      return cmdline.includes('hermes')
    } catch {
      // /proc not available (macOS) — fall back to ps. Use -o args= to inspect
      // the full command line, not just the process name.  -o comm= would return
      // "python3" for any Python process, creating false positives.
      try {
        const { execSync } = require('child_process')
        const out = execSync(`ps -p ${pid} -o args=`, { encoding: 'utf8', timeout: 2000 })

        return out.includes('hermes')
      } catch {
        return false
      }
    }
  }

  // Seed active-profile.json from the best available signal when the file does
  // not yet exist.  Runs exactly once (no-op once the file exists).  Priority:
  //   1. Legacy ~/.hermes/active_profile (explicit CLI choice via hermes profile use)
  //   2. Running gateway (gateway.pid with verified liveness + hermes identity)
  //   3. state.db heuristics (hybrid recency×size score picks the primary workspace)
  // The stored JSON includes _migrated:true so the renderer can optionally surface
  // a one-time notification that the profile was auto-detected.
  //
  // Decision logic lives in profile-migration.ts (pure + unit-tested). This wrapper
  // just wires Electron/Node fs into a MigrationDeps bag and delegates.
  function migrateActiveProfileIfMissing() {
    migrateActiveProfileIfMissingPure(DESKTOP_PROFILE_CONFIG_PATH, {
      legacyActivePath: path.join(HERMES_HOME, 'active_profile'),
      hermesHome: HERMES_HOME,
      profilesRoot: path.join(HERMES_HOME, 'profiles'),
      existsSync: p => fs.existsSync(p),
      readFileSync: (p, enc) => fs.readFileSync(p, enc),
      statSync: p => fs.statSync(p),
      readdirSync: (p, opts) => fs.readdirSync(p, opts as { withFileTypes: true }),
      isHermesProcess,
      now: () => Date.now(),
      writeJson: (target, decision) => {
        // Mirror writeActiveDesktopProfile's atomic-write + parent-dir-create
        // semantics so the migration produces a file indistinguishable from a
        // user-driven profile switch.
        fs.mkdirSync(path.dirname(target), { recursive: true })
        writeFileAtomic(target, JSON.stringify(decision, null, 2))
      },
      isValidProfileName: p => PROFILE_NAME_RE.test(p)
    })
  }

  // Options describing the current connection setup for `resolveProfileBackendRoute`.
  function profileRouteOptions(profile, request?) {
    const config = readDesktopConnectionConfig()
    const sshOverride = profileSshOverride(config, profile)
    const key = connectionScopeKey(profile) || primaryProfileKey()

    return {
      // A desktop profile can be only a client-side routing alias. Keep backend
      // endpoint filters in the SSH target's namespace (e.g. mara → default).
      backendProfile: sshOverride?.remoteProfile,
      globalRemote: globalRemoteActive(),
      primaryProfile: primaryProfileKey(),
      profileRemoteOverride: Boolean(profileRemoteOverride(config, profile) || sshOverride),
      // The primary profile's own backend resolves to a remote host (its
      // per-profile override, env, or global). Unknown sub-profiles on that
      // gateway must route THROUGH it, not spawn local backends (#88296).
      primaryRemoteActive: primaryBackendIsRemote(),
      // A stored per-profile entry (local or remote) — pins this profile to
      // its own backend; absent entries inherit the primary's remote.
      ownEntry: Boolean((config.profiles || {})[key]),
      isolatedBackend: getIsolatedBackend(),
      requestMethod: request?.method,
      requestPath: request?.path
    }
  }

  return {
    desktopProfilePreferences,
    validateDesktopProfileRoute,
    readActiveDesktopProfile,
    writeActiveDesktopProfile,
    migrateActiveProfileIfMissing,
    profileRouteOptions
  }
}
