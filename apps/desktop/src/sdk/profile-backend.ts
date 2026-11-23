import { $profiles } from '@/store/profile'
import { refreshProfileRemoteOverrides } from '@/store/profile-remote-override'
import { beginSwitchTransition, endSwitchTransition } from '@/store/switch-transition'

// The plugin SDK's host.setProfileBackend / host.clearProfileBackend: a desktop
// plugin runs one profile's backend on this computer (for example in a sandbox)
// and routes that profile to it. Main confirms every change natively and
// accepts a loopback address only (electron/profile-backend-ipc.ts); this side
// runs the plugin's hooks around the user's answer and shows the profile's
// "Starting sandbox…" / "Stopping sandbox…" state meanwhile.

export interface ProfileBackendAddress {
  /** The backend's access token. */
  token: string
  /** Its loopback base URL, e.g. http://127.0.0.1:47911. */
  url: string
}

export interface SetProfileBackendInput extends Partial<ProfileBackendAddress> {
  /** The calling plugin's id; the confirm names it. */
  plugin: string
  /**
   * Start the backend after the user agrees, resolving its address. The
   * profile stays on this computer's backend until then; a Cancel never runs
   * it, and if it throws nothing changes. Instead of `url` and `token`.
   */
  prepare?: () => ProfileBackendAddress | Promise<ProfileBackendAddress>
  profile: string
  /** One plain line the confirm shows as the plugin's own words. */
  purpose?: string
}

export interface ClearProfileBackendInput {
  /**
   * Runs after the user agrees and before the profile leaves the plugin's
   * backend (stop it there). If it throws, nothing changes.
   */
  beforeApply?: () => unknown
  plugin: string
  profile: string
}

/** What this build's setProfileBackend / clearProfileBackend support. */
export const profileBackendFeatures = Object.freeze({ beforeApply: true, prepare: true, purpose: true })

function profileBackendBridge() {
  const desktop = window.hermesDesktop

  if (
    !desktop?.setProfileBackend ||
    !desktop.confirmSetProfileBackend ||
    !desktop.clearProfileBackend ||
    !desktop.confirmClearProfileBackend
  ) {
    throw new Error('This build of Hermes Desktop cannot run a profile through a plugin.')
  }

  return desktop as Required<
    Pick<
      typeof desktop,
      'clearProfileBackend' | 'confirmClearProfileBackend' | 'confirmSetProfileBackend' | 'setProfileBackend'
    >
  >
}

function profileBackendTarget(input: { plugin: string; profile: string }) {
  return { plugin: String(input?.plugin ?? '').trim(), profile: String(input?.profile ?? '').trim() }
}

// The rail's badges and the transition hooks read each profile's route.
function refreshProfileRoutes(): Promise<void> {
  return refreshProfileRemoteOverrides($profiles.get().map(profile => profile.name)).catch(() => undefined)
}

/**
 * Route a profile to a backend this plugin runs on this computer, after the
 * user confirms. Resolves `{ applied: false }` when the user declines.
 */
export async function setProfileBackend(input: SetProfileBackendInput): Promise<{ applied: boolean }> {
  const bridge = profileBackendBridge()
  const purpose = typeof input?.purpose === 'string' ? input.purpose.trim() : ''
  const target = { ...profileBackendTarget(input), ...(purpose ? { purpose } : {}) }
  const prepare = typeof input?.prepare === 'function' ? input.prepare : null
  let grant: string | undefined

  if (prepare) {
    const answer = await bridge.confirmSetProfileBackend(target)

    if (!answer?.confirmed || !answer.grant) {
      return { applied: false }
    }

    grant = answer.grant
    beginSwitchTransition(target.profile, 'starting')
  }

  let applied = false

  try {
    const address = prepare ? await prepare() : input
    const token = String(address?.token ?? '')
    const url = String(address?.url ?? '').trim()

    applied = Boolean((await bridge.setProfileBackend({ ...target, ...(grant ? { grant } : {}), token, url }))?.applied)
    await refreshProfileRoutes()
  } finally {
    if (grant) {
      endSwitchTransition(target.profile, applied)
    }
  }

  return { applied }
}

/**
 * Route a profile this plugin runs back to this computer's backend, after the
 * user confirms. Resolves `{ applied: false }` when the user declines or the
 * plugin does not run that profile.
 */
export async function clearProfileBackend(input: ClearProfileBackendInput): Promise<{ applied: boolean }> {
  const bridge = profileBackendBridge()
  const target = profileBackendTarget(input)
  const beforeApply = typeof input?.beforeApply === 'function' ? input.beforeApply : null
  let grant: string | undefined

  if (beforeApply) {
    const answer = await bridge.confirmClearProfileBackend(target)

    if (!answer?.confirmed || !answer.grant) {
      return { applied: false }
    }

    grant = answer.grant
    beginSwitchTransition(target.profile, 'stopping')
  }

  let applied = false

  try {
    await beforeApply?.()
    applied = Boolean((await bridge.clearProfileBackend({ ...target, ...(grant ? { grant } : {}) }))?.applied)
    await refreshProfileRoutes()
  } finally {
    if (grant) {
      endSwitchTransition(target.profile, applied)
    }
  }

  return { applied }
}

/** The entries `host` carries for a plugin that runs a profile's backend on this computer. */
export const profileBackendHost = { clearProfileBackend, profileBackendFeatures, setProfileBackend }
