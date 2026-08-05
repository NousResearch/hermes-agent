import { translateNow } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { newSessionInProfile, normalizeProfileKey, refreshProfiles } from '@/store/profile'

import { requestComposerFocus, requestComposerInsert } from '../chat/composer/focus'

interface DesktopDeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

const PROFILE_NAME_RE = /^(?:default|[a-z0-9][a-z0-9_-]{0,63})$/
let latestDeliveryGeneration = 0

export async function handleDesktopDeepLink(payload: DesktopDeepLinkPayload | null | undefined): Promise<void> {
  if (!payload?.name) {
    return
  }

  if (payload.kind === 'blueprint') {
    latestDeliveryGeneration += 1

    const slots = Object.entries(payload.params || {})
      .map(([key, value]) => {
        const stringValue = /\s/.test(value) ? `"${value.replace(/"/g, '\\"')}"` : value

        return `${key}=${stringValue}`
      })
      .join(' ')

    const command = `/blueprint ${payload.name}${slots ? ' ' + slots : ''}`
    requestComposerInsert(command, { mode: 'block', target: 'main' })
    requestComposerFocus('main')

    return
  }

  if (payload.kind !== 'profile' || payload.params?.new !== '1' || !PROFILE_NAME_RE.test(payload.name)) {
    return
  }

  const deliveryGeneration = ++latestDeliveryGeneration
  let profiles

  try {
    profiles = await refreshProfiles()
  } catch (error) {
    if (deliveryGeneration === latestDeliveryGeneration) {
      notifyError(error, translateNow('desktop.setProfileFailed'))
    }

    return
  }

  // Profile discovery is asynchronous. A newer external link owns the user's
  // navigation intent, so a stale lookup must not replace its draft or error.
  if (deliveryGeneration !== latestDeliveryGeneration) {
    return
  }

  const target = normalizeProfileKey(payload.name)
  const installed = profiles.some(profile => normalizeProfileKey(profile.name) === target)

  if (!installed) {
    const available = profiles.map(profile => profile.name).join(', ') || '—'

    notify({
      id: `deep-link-profile-missing:${target}`,
      kind: 'error',
      title: translateNow('desktop.unknownProfile'),
      message: translateNow('desktop.noProfileNamed', target, available)
    })

    return
  }

  // This is intentionally the live per-session route. It does not call the
  // persistent profile setter and never changes the CLI's sticky default.
  newSessionInProfile(target)
  requestComposerFocus('main')
}
