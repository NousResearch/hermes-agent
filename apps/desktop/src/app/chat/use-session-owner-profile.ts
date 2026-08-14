import { useStore } from '@nanostores/react'

import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

/**
 * The profile that owns the chat surface currently rendering this component.
 *
 * The renderer adopts the connected backend profile before the gateway opens,
 * and session tiles are persisted and swapped per active profile. Do not infer
 * ownership from the bounded recent-session cache: the parent may be older than
 * that cache, while the live gateway remains the source of truth.
 */
export function useSessionOwnerProfile(): string {
  return normalizeProfileKey(useStore($activeGatewayProfile))
}
