import { atom } from 'nanostores'

/**
 * Pending `hermes://bot/<profile>` open request, set by the deep-link listener
 * and consumed by the hermes-bots plugin. Null means no request. The payload
 * is a bot NAME — identity is resolved at open time through the same
 * "Bot Chat" title registry the roster row click uses, never a session id.
 */
export const $pendingDeepLinkBot = atom<string | null>(null)

/** Validate a bot deep link's target into a pending open, or reject silently. */
export function requestBotChatFromDeepLink(rawProfile: string | undefined): boolean {
  const profile = (rawProfile || '').trim()

  if (!profile) {
    return false
  }

  $pendingDeepLinkBot.set(profile)

  return true
}
