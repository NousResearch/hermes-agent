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
