import { profileNameFromPath } from './profile-delete-routing'

export interface ProfileRenameRequest {
  body?: unknown
  connectionId?: unknown
  method?: unknown
  path?: unknown
  profile?: unknown
}

export interface ProfileRename {
  newName: string
  oldName: string
}

export interface ProfileRenameLifecycleDeps {
  isValidProfileName: (profile: string) => boolean
  primaryProfileKey: () => string
  reloadPrimaryWindow: () => void
  restartPrimaryBackend: () => Promise<void>
  teardownPoolBackendAndWait: (profile: string) => Promise<void>
  teardownPrimaryBackendAndWait: () => Promise<void>
  writeActiveDesktopProfile: (profile: string) => void
}

export interface ProfileRenameLifecycle {
  complete: () => Promise<void>
  kind: 'pool' | 'primary'
  rename: ProfileRename
  rollback: () => Promise<void>
  routeProfile: null
}

export interface ConnectionScopedProfileRenameDeps<T> {
  acquire: (profile: string) => () => void
  connectionKind: (connectionId: string) => string
  dispatch: (routeProfile: null) => Promise<T>
  isValidProfileName: (profile: string) => boolean
  prepareLocal: (request: ProfileRenameRequest) => Promise<ProfileRenameLifecycle | null>
  teardownConnection: (connectionId: string, profile: string) => Promise<void>
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

export async function dispatchConnectionScopedProfileRename<T>(
  request: ProfileRenameRequest,
  deps: ConnectionScopedProfileRenameDeps<T>
): Promise<T> {
  const rename = profileRenameFromRequest(request)
  const connectionId = String(request.connectionId ?? '').trim()
  const logicalProfile = String(request.profile ?? '').trim() || rename?.oldName || ''

  if (!connectionId || !rename) {
    throw new Error('Connection-scoped profile rename requires a connection and old/new profile names.')
  }

  if (!deps.isValidProfileName(rename.oldName) || !deps.isValidProfileName(rename.newName)) {
    throw new Error('Invalid profile rename.')
  }

  const connectionKind = deps.connectionKind(connectionId)
  const release = deps.acquire(rename.oldName)

  try {
    const lifecycle = connectionKind === 'local' ? await deps.prepareLocal(request) : null

    if (connectionKind === 'local' && !lifecycle) {
      throw new Error('Unable to prepare local profile rename.')
    }

    if (connectionKind !== 'local') {
      await deps.teardownConnection(connectionId, logicalProfile)
    }

    let response: T

    try {
      response = await deps.dispatch(lifecycle?.routeProfile ?? null)
    } catch (error) {
      await lifecycle?.rollback()
      throw error
    }

    await lifecycle?.complete()

    return response
  } finally {
    release()
  }
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

  // Make `default` the temporary primary before stopping the old backend.
  // Concurrent primary requests then share the temporary connection instead
  // of respawning the old profile and recreating its directory mid-rename.
  deps.writeActiveDesktopProfile('default')

  try {
    await deps.teardownPrimaryBackendAndWait()
  } catch (error) {
    deps.writeActiveDesktopProfile(rename.oldName)

    try {
      await deps.restartPrimaryBackend()
    } catch {
      // Preserve the teardown error that prevented the rename from starting.
    }

    throw error
  }

  return {
    complete: async () => {
      deps.writeActiveDesktopProfile(rename.newName)

      try {
        await deps.teardownPrimaryBackendAndWait()
      } finally {
        deps.reloadPrimaryWindow()
      }
    },
    kind: 'primary',
    rename,
    rollback: async () => {
      deps.writeActiveDesktopProfile(rename.oldName)

      try {
        await deps.teardownPrimaryBackendAndWait()
      } finally {
        await deps.restartPrimaryBackend()
      }
    },
    routeProfile: null
  }
}
