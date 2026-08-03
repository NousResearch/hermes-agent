/**
 * Command-palette contribution surface — `palette` data contributions become
 * rows in the ⌘K root list, same schema as every other area. Contributions
 * with an `action` id render that action's live keybind as their hotkey hint.
 */

import { useContributions } from '@/contrib/react/use-contributions'
import type { IconComponent } from '@/lib/icons'

export const PALETTE_AREA = 'palette'

/** Payload of a `palette` data contribution. */
export interface PaletteContribution {
  id: string
  label: string
  /** Keybind action id — its live combo renders as the hotkey hint. */
  action?: string
  icon?: IconComponent
  keywords?: string[]
  run: () => void
}

/** Contributed palette rows, with stable render keys. */
export function usePaletteContributions(): Array<PaletteContribution & { key: string }> {
  return useContributions(PALETTE_AREA)
    .map(c => ({ key: `${c.source ?? 'core'}:${c.id}`, ...(c.data as PaletteContribution) }))
    .filter(item => Boolean(item.label && item.run))
}
