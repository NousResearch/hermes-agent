import { isValidElement, type ReactElement, type ReactNode, useMemo } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib/react/use-contributions'

/**
 * Model menu row decorations — a plugin's per-model mark inside the native
 * model catalog menu (the composer's model pill menu and every other surface
 * that renders `ModelCatalogMenu`).
 *
 *   data kind (`data`):  modelMenu.row  (ModelMenuRowContribution)
 *
 * Core keeps the row: its markup, name, chips, star, submenu and what a click
 * means. A contribution only fills two fixed slots — a leading icon before the
 * name and a trailing badge after core's own chips — so several plugins can
 * register at once without fighting over the row.
 */
export const MODEL_MENU_ROW_AREA = 'modelMenu.row'

/** What a row decorator gets to branch on. Deliberately small: every field
 *  here is a standing compatibility promise to the plugins using it. */
export interface ModelMenuRowContext {
  /** The display name core paints on the row (`Opus 4.5`). */
  label: string
  /** The model id the row commits (the base id when a `-fast` sibling shares it). */
  model: string
  /** The provider slug the row belongs to (`anthropic`, `openrouter`, …). */
  provider: string
}

export interface ModelMenuRowDecoration {
  /** Plain text chip after core's own chips (`$3/M`, `NEW`). */
  badge?: string
  /** Leading mark drawn in a fixed 1rem box before the name: an element
   *  (`<img>`, `<svg>`, a component) or short text (an emoji, a monogram). */
  icon?: ReactNode
}

/** Payload of a `modelMenu.row` data contribution. Return the decoration for
 *  this row, or `null` for rows you have nothing to say about. */
export interface ModelMenuRowContribution {
  decorate: (row: ModelMenuRowContext) => ModelMenuRowDecoration | null
}

const usableIcon = (icon: unknown): icon is ReactNode =>
  isValidElement(icon) || (typeof icon === 'string' && icon.trim() !== '')

const usableBadge = (badge: unknown): badge is string => typeof badge === 'string' && badge.trim() !== ''

/** The row's decoration: per slot, the first contribution (registry order)
 *  that supplies a usable value wins, so an icon plugin and a badge plugin
 *  compose. A decorator that throws or returns garbage declines — a broken
 *  plugin can never take the menu down. */
export function useModelMenuRowDecoration({ label, model, provider }: ModelMenuRowContext): ModelMenuRowDecoration {
  const contributions = useContributions(MODEL_MENU_ROW_AREA)

  return useMemo(() => {
    const result: ModelMenuRowDecoration = {}

    for (const contribution of contributions) {
      if (result.icon !== undefined && result.badge !== undefined) {
        break
      }

      try {
        const { badge, icon } =
          (contribution.data as ModelMenuRowContribution | undefined)?.decorate?.({ label, model, provider }) ?? {}

        if (result.icon === undefined && usableIcon(icon)) {
          result.icon = icon
        }

        if (result.badge === undefined && usableBadge(badge)) {
          result.badge = badge.trim()
        }
      } catch {
        // Decline on throw: the next contribution, then the bare row, wins.
      }
    }

    return result
  }, [contributions, label, model, provider])
}

/** The fixed leading slot. Its own boundary: a plugin component that throws
 *  while rendering blanks only its icon, never the row or the menu. */
export function ModelMenuRowIcon({ icon }: { icon: ReactNode }): ReactElement {
  return (
    <span
      aria-hidden="true"
      className="grid size-4 shrink-0 place-items-center overflow-hidden text-[0.75rem] leading-none [&>img]:size-full [&>img]:object-contain [&>svg]:size-full"
      data-slot="model-menu-row-icon"
    >
      <ErrorBoundary fallback={() => null} label={`contrib:${MODEL_MENU_ROW_AREA}`}>
        {icon}
      </ErrorBoundary>
    </span>
  )
}
