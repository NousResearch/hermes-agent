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
  /**
   * Muted text after the label — the live state the row acts on. A function
   * because contributions register once at boot while that state keeps moving;
   * the palette re-reads it on open.
   */
  detail?: () => string
}

/** Contributed palette rows, with stable render keys. */
export function usePaletteContributions(): Array<PaletteContribution & { key: string }> {
  return useContributions(PALETTE_AREA)
    .map(c => ({ key: `${c.source ?? 'core'}:${c.id}`, ...(c.data as PaletteContribution) }))
    .filter(item => Boolean(item.label && item.run))
}

/**
 * A binary setting as one palette row: `Toggle status bar` with the live state
 * trailing in muted text. The verb says what the row does, the detail says
 * which way it will go — neither alone is enough.
 */
export function paletteToggle(
  spec: Omit<PaletteContribution, 'detail' | 'run'> & {
    get: () => boolean
    set: (enabled: boolean) => void
  }
) {
  const { get, keywords = [], set, ...rest } = spec

  const data: PaletteContribution = {
    ...rest,
    detail: () => (get() ? 'on' : 'off'),
    keywords: [...keywords, 'on', 'off', 'enable', 'disable'],
    run: () => set(!get())
  }

  return { id: data.id, area: PALETTE_AREA, data }
}
