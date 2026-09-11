import { readKey, writeKey } from '@/lib/storage'
import { notify } from '@/store/notifications'

const FIRST_HIDE_SEEN_KEY = 'zone-hide-first-time-toast-seen'

/**
 * Show a one-time educational toast the first time a user minimizes a zone,
 * telling them how to restore it with ⌘B. Addresses #107927 — zone-level hide
 * has no visible restore affordance, so discoverability relies on keyboard
 * knowledge.
 */
export function maybeShowZoneHideToast() {
  const seen = readKey(FIRST_HIDE_SEEN_KEY)

  if (seen === 'true') {
    return
  }

  notify({
    kind: 'info',
    message: 'Sidebar hidden — press ⌘B to bring it back',
    durationMs: 8_000,
    placement: 'default'
  })

  writeKey(FIRST_HIDE_SEEN_KEY, 'true')
}
