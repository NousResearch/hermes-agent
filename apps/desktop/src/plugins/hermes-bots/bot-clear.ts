/** Clear one bot's existing canonical Bot Chat without changing its identity. */

import { host, queryClient } from '@hermes/plugin-sdk'

import { isCanonicalChatOnScreen, openBotCanonicalChat } from './canonical-chat'
import { ROSTER_KEY } from './data'
import { backendTargetProfile, botConnectionRoute, requestForBot } from './routing'
import type { RosterRow } from './types'

function isMissingClearCapability(error: unknown): boolean {
  const message = String((error as { message?: unknown })?.message || error || '')

  return /(?:method not found|-32601|no handler for|unknown method|unsupported rpc)/i.test(message)
}

/**
 * Ask the owning backend to clear its named Bot Chat registry row. The RPC
 * deliberately receives only the profile: the backend owns the exact
 * `Bot Chat` title lookup, so no stored session-id pointer can creep back in.
 */
export async function clearBotCanonicalChat(bot: RosterRow): Promise<void> {
  const focusedBefore = String(host.state.focusedStoredSessionId?.get?.() || '')
  const wasFocused = isCanonicalChatOnScreen(bot, focusedBefore)
  const route = botConnectionRoute(bot)

  try {
    await requestForBot(bot, 'session.clear_bot_chat', {
      profile: backendTargetProfile(route, bot.name)
    })
  } catch (error) {
    if (isMissingClearCapability(error)) {
      throw new Error('This gateway cannot clear Bot Chat yet. Update the gateway, then try again.')
    }

    throw error
  }

  await queryClient.invalidateQueries({ queryKey: ROSTER_KEY })

  // Clearing a background bot must not steal the workspace. If the user also
  // switched away while the RPC was in flight, leave their new focus alone.
  if (wasFocused && isCanonicalChatOnScreen(bot, host.state.focusedStoredSessionId?.get?.())) {
    await openBotCanonicalChat(bot, () => isCanonicalChatOnScreen(bot, host.state.focusedStoredSessionId?.get?.()))
  }
}
