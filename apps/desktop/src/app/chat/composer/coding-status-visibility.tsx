import type { ReactNode } from 'react'

import type { HermesConfigRecord } from '@/types/hermes'

interface CodingStatusVisibilityProps {
  children: ReactNode
  config: HermesConfigRecord | undefined
}

/**
 * The coding status row is opt-out so older or partially loaded config records
 * preserve the existing Desktop behavior until the setting is explicitly off.
 */
export function shouldShowCodingStatus(config: HermesConfigRecord | undefined): boolean {
  const display = config?.display

  if (!display || typeof display !== 'object' || Array.isArray(display)) {
    return true
  }

  return (display as Record<string, unknown>).show_coding_status !== false
}

export function CodingStatusVisibility({ children, config }: CodingStatusVisibilityProps) {
  return shouldShowCodingStatus(config) ? children : null
}
