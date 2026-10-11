import { profileNameFromPath } from './profile-delete-routing'

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

  // Make `default` the temporary primary before stopping the old backend.
  // Concurrent primary requests then share the temporary connection instead
  // of respawning the old profile and recreating its directory mid-rename.
  deps.writeActiveDesktopProfile('default')

  try {
    // The old name can also own an explicit-local pool backend
    // (`conn:local::<name>`); it holds the same profile home open.
    await Promise.all([deps.teardownPrimaryBackendAndWait(), deps.teardownPoolBackendAndWait(rename.oldName)])
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

export interface ConnectionScopedProfileRenameRequest extends ProfileRenameRequest {
  connectionId?: unknown
  profile?: unknown
}

export interface ConnectionScopedProfileRenameDeps<T> {
  acquire: (profile: string) => () => void
  /** Post-success bookkeeping. A failure here must not roll back a rename the backend already made. */
  afterResponse: (result: T) => void
  connectionKind: (connectionId: string) => string
  dispatch: (routeProfile: null) => Promise<T>
  isValidProfileName: (profile: string) => boolean
  logRollbackError: (error: unknown) => void
  prepareLocal: (request: ProfileRenameRequest) => Promise<null | ProfileRenameLifecycle>
  teardownConnection: (connectionId: string, profile: string) => Promise<void>
}

/**
 * Run a rename PATCH pinned to a registry connection under the same gate as
 * dispatchConnectionScopedProfileDelete. The old name's backends stop first and
 * the PATCH dispatches through a null route profile: dialling the old name
 * would trip the gate this call holds, which is how a registry-pinned rename
 * used to fail with `Profile "<name>" is being deleted.` A local rename also
 * re-homes the primary on success and restores it on failure.
 */
export async function dispatchConnectionScopedProfileRename<T>(
  request: ConnectionScopedProfileRenameRequest,
  deps: ConnectionScopedProfileRenameDeps<T>
): Promise<T> {
  const rename = profileRenameFromRequest(request)
  const connectionId = String(request.connectionId ?? '').trim()

  if (!rename || !connectionId) {
    throw new Error('Connection-scoped profile rename requires a connection and profile.')
  }

  for (const profile of [rename.oldName, rename.newName]) {
    if (!deps.isValidProfileName(profile)) {
      throw new Error(`Invalid profile name: ${profile}`)
    }
  }

  const release = deps.acquire(rename.oldName)

  try {
    let lifecycle: null | ProfileRenameLifecycle = null

    if (deps.connectionKind(connectionId) === 'local') {
      lifecycle = await deps.prepareLocal(request)
    } else {
      await deps.teardownConnection(connectionId, String(request.profile ?? '').trim() || rename.oldName)
    }

    let result: T

    try {
      result = await deps.dispatch(null)
    } catch (error) {
      try {
        await lifecycle?.rollback()
      } catch (rollbackError) {
        deps.logRollbackError(rollbackError)
      }

      throw error
    }

    try {
      deps.afterResponse(result)
    } finally {
      await lifecycle?.complete()
    }

    return result
  } finally {
    release()
  }
}
