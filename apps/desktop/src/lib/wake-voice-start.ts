import { requestVoiceConversationStart } from '@/store/composer'
import { $activeGatewayProfile, activateOnCurrentSource, normalizeProfileKey, pinNewChatProfile } from '@/store/profile'

export interface WakeVoicePayload {
  profile?: null | string
  start_new_session?: boolean
}

let startSequence = 0

/** A detected phrase owns its profile before the microphone or Codex credentials are acquired. */
export async function startWakeVoiceConversation(
  payload: WakeVoicePayload | undefined,
  startFreshSession: () => void
): Promise<void> {
  const sequence = ++startSequence
  const target = normalizeProfileKey(payload?.profile?.trim() || $activeGatewayProfile.get())

  // Use the same connection-aware route as a profile pick, including local
  // profile overrides. Await failures; never start voice on the previous login.
  await activateOnCurrentSource(target)

  if (sequence !== startSequence || normalizeProfileKey($activeGatewayProfile.get()) !== target) {
    return
  }

  if (payload?.start_new_session !== false) {
    pinNewChatProfile(target)
    startFreshSession()
  }

  requestVoiceConversationStart()
}
