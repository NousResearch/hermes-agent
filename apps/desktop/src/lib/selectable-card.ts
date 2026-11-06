import { cn } from '@/lib/utils'

export interface SelectableCardState {
  /** Currently selected / active — the strongest emphasis. */
  active?: boolean
  /**
   * Configured / installed / "you have this" — solid surface + border. When
   * false the card renders muted (transparent, dimmed) until hovered, so the
   * eye lands on what you already have. Ignored when `active` is set.
   */
  prominent?: boolean
}

/**
 * Shared emphasis for selectable list cards across settings surfaces (theme
 * picker, pet picker, Marketplace results, provider rows…). Three tiers:
 * active > prominent > muted. Keeps the "installed = solid, not-installed =
 * quiet" pattern consistent everywhere instead of each picker rolling its own.
 *
 * Callers own layout (padding, flex, width); this owns only border + surface.
 */
export function selectableCardClass({ active, prominent }: SelectableCardState): string {
  return cn(
    'rounded-lg border transition-colors',
    active
      ? 'border-primary bg-primary/[0.06] ring-2 ring-primary/20'
      : prominent
        ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) hover:bg-(--chrome-action-hover)'
        : 'border-transparent bg-transparent text-(--ui-text-tertiary) hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-bg-quinary)'
  )
}
