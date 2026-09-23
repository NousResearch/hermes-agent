import { decideProfileDeleteAction, profileNameFromDeleteRequest } from './profile-delete-routing'
import { prepareProfileRenameLifecycle } from './profile-rename-routing'

export interface DesktopProfileMutationRuntimeDeps {
  profileNameRe: RegExp
  primaryProfileKey: any
  writeActiveDesktopProfile: any
  teardownPrimaryBackendAndWait: any
  teardownPoolBackendAndWait: any
  getMainWindow: any
  startHermes: any
}

export function createDesktopProfileMutationRuntime(deps: DesktopProfileMutationRuntimeDeps) {
  const PROFILE_NAME_RE = deps.profileNameRe

  const {
    primaryProfileKey,
    writeActiveDesktopProfile,
    teardownPrimaryBackendAndWait,
    teardownPoolBackendAndWait,
    getMainWindow,
    startHermes
  } = deps

  // Returns the profile name whose backend was torn down, or null when the
  // request is not a profile-delete.  The caller uses this to skip ensureBackend
  // for the just-torn-down profile — otherwise ensureBackend respawns a pool
  // backend whose ensure_hermes_home() recreates the deleted profile directory.
  //
  // The routing *decision* (which branch fires, what profile name gets
  // returned) lives in the pure decideProfileDeleteAction() in
  // profile-delete-routing.ts; this function only performs the side effects
  // that decision calls for.
  async function prepareProfileDeleteRequest(request) {
    const profile = profileNameFromDeleteRequest(request)

    const decision = decideProfileDeleteAction(profile, {
      isDefaultProfile: p => p === 'default',
      isValidProfileName: p => PROFILE_NAME_RE.test(p),
      primaryProfileKey
    })

    if (decision.action === 'noop') {
      return null
    }

    if (decision.action === 'teardown-primary') {
      writeActiveDesktopProfile('default')
      await Promise.all([teardownPrimaryBackendAndWait(), teardownPoolBackendAndWait(decision.profile)])

      return decision.profile
    }

    await teardownPoolBackendAndWait(decision.profile)

    return decision.profile
  }

  async function prepareProfileRenameRequest(request) {
    return prepareProfileRenameLifecycle(request, {
      isValidProfileName: profile => PROFILE_NAME_RE.test(profile),
      primaryProfileKey,
      reloadPrimaryWindow: () => {
        getMainWindow()?.reload()
      },
      restartPrimaryBackend: async () => {
        await startHermes()
      },
      teardownPoolBackendAndWait,
      teardownPrimaryBackendAndWait,
      writeActiveDesktopProfile: profile => {
        writeActiveDesktopProfile(profile)
      }
    })
  }

  return { prepareProfileDeleteRequest, prepareProfileRenameRequest }
}
