// The native confirm behind the plugin SDK's host.setProfileBackend /
// host.clearProfileBackend. Any desktop plugin can make that call and main
// cannot tell plugins apart, so the dialog is the user's only check that a
// plugin is not quietly moving a profile's chats. It names the plugin and the
// profile and says where the chats will run in plain words; the exact address
// is only a detail line. A plugin may add a short purpose, shown as the
// plugin's own words, never as the app's.

export interface ProfileBackendConfirmCopy {
  buttons: [string, string]
  detail: string
  message: string
}

export const PROFILE_BACKEND_PURPOSE_MAX = 160

/**
 * A plugin's purpose line: one line of plain text, at most
 * PROFILE_BACKEND_PURPOSE_MAX characters, or null. Control characters and the
 * quotes the dialog wraps it in are dropped, so it cannot fake a line of the
 * app's own copy.
 */
export function profileBackendPurpose(raw: unknown): null | string {
  if (typeof raw !== 'string') {
    return null
  }

  const text = raw
    // eslint-disable-next-line no-control-regex -- matching control characters is the point
    .replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ')
    .replace(/[“”"]/g, "'")
    .replace(/\s+/g, ' ')
    .trim()

  if (!text) {
    return null
  }

  return text.length > PROFILE_BACKEND_PURPOSE_MAX
    ? `${text.slice(0, PROFILE_BACKEND_PURPOSE_MAX - 1).trimEnd()}…`
    : text
}

/**
 * The confirm for host.setProfileBackend. `url` is the backend's address when
 * the plugin already runs it; with `prepare` the plugin starts it only after
 * the user agrees, so there is no address yet. Main accepts a backend on this
 * computer only, either way.
 */
export function setProfileBackendConfirm(input: {
  plugin: string
  profile: string
  purpose?: unknown
  url?: string
}): ProfileBackendConfirmCopy {
  const { plugin, profile, url } = input
  const purpose = profileBackendPurpose(input.purpose)
  const verb = url ? 'runs' : 'starts'

  const lines = [
    `Chats in “${profile}” will run in a backend that ${plugin} ${verb} on this computer, instead of Hermes's own backend.`,
    ...(purpose ? [`${plugin} says: “${purpose}”`] : []),
    ...(url ? [`Address: ${url}`] : [])
  ]

  return {
    buttons: [`Switch to ${plugin}`, 'Cancel'],
    detail: lines.join('\n\n'),
    message: `Let the ${plugin} plugin run the “${profile}” profile?`
  }
}

/** The confirm for host.clearProfileBackend. */
export function clearProfileBackendConfirm(input: { plugin: string; profile: string }): ProfileBackendConfirmCopy {
  const { plugin, profile } = input

  return {
    buttons: ['Switch back', 'Cancel'],
    detail: `Chats in “${profile}” will run in Hermes's own backend on this computer again.`,
    message: `Stop the ${plugin} plugin from running the “${profile}” profile?`
  }
}
