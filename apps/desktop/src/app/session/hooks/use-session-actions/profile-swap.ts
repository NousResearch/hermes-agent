// #81817 / #79406: during a pending gateway swap the quick-create selection
// ($newChatProfile) is already cleared while the swap is still opening the
// TARGET profile's backend. Reading only the still-live atoms at that moment
// binds a new chat to (and inherits the cwd of) the profile the user just
// left, so the pending swap target is the visible intent and wins the
// fallback.
import { $activeGatewayProfile, $gatewaySwapTarget, normalizeProfileKey } from '@/store/profile'

// The new-chat profile: any explicit selection, else the pending swap target,
// else the live gateway profile.
export function resolveChatProfile(explicit: null | string | undefined): string {
  return normalizeProfileKey(explicit?.trim() || $gatewaySwapTarget.get() || $activeGatewayProfile.get())
}
