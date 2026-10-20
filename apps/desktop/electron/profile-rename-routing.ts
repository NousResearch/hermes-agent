import { type ProfileRouteOptions, resolveProfileBackendRoute } from './connection-config'
import { profileNameFromDeleteRequest, profileNameFromPath } from './profile-delete-routing'

export interface ProfileRenameRequest {
  body?: unknown
  method?: unknown
  path?: unknown
}

export interface ProfileRename {
  newName: string
  oldName: string
}

export interface ProfileRenameLifecycleDeps {
  isValidProfileName: (profile: string) => boolean
  primaryProfileKey: () => string
  readActiveDesktopProfile: () => null | string
  reloadPrimaryWindow: () => void
  restartPrimaryBackend: () => Promise<void>
  teardownPoolBackendAndWait: (profile: string) => Promise<void>
  teardownPrimaryBackendAndWait: () => Promise<void>
  writeActiveDesktopProfile: (profile: string) => void
}

interface ProfileMutationStartupDeps<T> {
  dispatch: () => Promise<T>
  local: boolean
  primaryProfileKey: () => string
  readActiveDesktopProfile: () => null | string
  writeActiveDesktopProfile: (profile: string) => void
}

export function profileMutationIsLocal(profile: unknown, options: ProfileRouteOptions): boolean {
  const route = resolveProfileBackendRoute(profile, options)

  return route.backend === 'pool'
    ? !options.profileRemoteOverride
    : !options.primaryRemoteActive && !options.globalRemote
}

/** A live pooled workspace can also be the next-launch choice. Changing its
 * name/home must update that choice without re-homing the primary backend. */
export async function dispatchProfileMutationWithStartupPreference<T>(
  request: ProfileRenameRequest,
  deps: ProfileMutationStartupDeps<T>
): Promise<T> {
  const rename = profileRenameFromRequest(request)
  const oldName = rename?.oldName ?? profileNameFromDeleteRequest(request)
  const remembered = deps.local && oldName && oldName !== 'default' && deps.readActiveDesktopProfile() === oldName
  const rehomesRememberedPrimary = remembered && deps.primaryProfileKey() === oldName

  try {
    const result = await deps.dispatch()

    if (remembered && deps.readActiveDesktopProfile() === oldName) {
      deps.writeActiveDesktopProfile(rename?.newName ?? 'default')
    }

    return result
  } catch (error) {
    // Primary deletion temporarily uses default to avoid respawning the home
    // being removed. Keep a failed mutation from discarding the saved choice.
    if (rehomesRememberedPrimary && deps.readActiveDesktopProfile() === 'default') {
      deps.writeActiveDesktopProfile(oldName)
    }

    throw error
  }
}

export interface ProfileRenameLifecycle {
  complete: () => Promise<void>
  kind: 'pool' | 'primary'
  rename: ProfileRename
  rollback: () => Promise<void>
  routeProfile: null
}

function parseJsonBody(body: unknown): Record<string, unknown> {
  if (body == null || body === '') {
    return {}
  }

  if (typeof body === 'object' && !Array.isArray(body)) {
    return body as Record<string, unknown>
  }

  try {
    const parsed = JSON.parse(String(body))

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

export function profileRenameFromRequest(request: ProfileRenameRequest | null | undefined): ProfileRename | null {
  if (!request || String(request.method || 'GET').toUpperCase() !== 'PATCH') {
    return null
  }

  const oldName = profileNameFromPath(request.path)

  if (!oldName || oldName === 'default') {
    return null
  }

  const body = parseJsonBody(request.body)

  const newName = String(body.new_name || '')
    .trim()
    .toLowerCase()

  if (!newName || newName === 'default') {
    return null
  }

  return { newName, oldName }
}

export async function prepareProfileRenameLifecycle(
  request: ProfileRenameRequest | null | undefined,
  deps: ProfileRenameLifecycleDeps
): Promise<ProfileRenameLifecycle | null> {
  const rename = profileRenameFromRequest(request)

  if (!rename || !deps.isValidProfileName(rename.oldName) || !deps.isValidProfileName(rename.newName)) {
    return null
  }

  if (rename.oldName !== deps.primaryProfileKey()) {
    await deps.teardownPoolBackendAndWait(rename.oldName)

    return {
      complete: async () => {},
      kind: 'pool',
      rename,
      rollback: async () => {},
      routeProfile: null
    }
  }

  // Re-home through the remembered workspace, using default temporarily only
  // when the renamed primary is itself remembered. Concurrent requests must
  // not respawn the old profile and recreate its directory mid-rename.
  const remembersPrimary = deps.readActiveDesktopProfile() === rename.oldName

  if (remembersPrimary) {
    deps.writeActiveDesktopProfile('default')
  }

  try {
    await deps.teardownPrimaryBackendAndWait()
  } catch (error) {
    if (remembersPrimary && deps.readActiveDesktopProfile() === 'default') {
      deps.writeActiveDesktopProfile(rename.oldName)
    }

    try {
      await deps.restartPrimaryBackend()
    } catch {
      // Preserve the teardown error that prevented the rename from starting.
    }

    throw error
  }

  return {
    complete: async () => {
      if (remembersPrimary && deps.readActiveDesktopProfile() === 'default') {
        deps.writeActiveDesktopProfile(rename.newName)
      }

      try {
        await deps.teardownPrimaryBackendAndWait()
      } finally {
        deps.reloadPrimaryWindow()
      }
    },
    kind: 'primary',
    rename,
    rollback: async () => {
      if (remembersPrimary && deps.readActiveDesktopProfile() === 'default') {
        deps.writeActiveDesktopProfile(rename.oldName)
      }

      try {
        await deps.teardownPrimaryBackendAndWait()
      } finally {
        await deps.restartPrimaryBackend()
      }
    },
    routeProfile: null
  }
}
