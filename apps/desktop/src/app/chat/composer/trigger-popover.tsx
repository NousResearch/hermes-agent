import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { Fragment, useEffect, useRef } from 'react'

import { referenceKind, referenceStyle } from '@/components/assistant-ui/reference-kinds'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { COMPLETION_DRAWER_BELOW_CLASS, COMPLETION_DRAWER_CLASS, CompletionDrawerEmpty } from './completion-drawer'
import type { DirectiveScope } from './text-utils'

interface RowMeta {
  display?: string
  group?: string
  meta?: string
}

/** The kind a row represents, for its icon. `@` rows carry it as the item type;
 *  `/` rows carry it as the completion group (Skills / Themes / Commands). */
function rowKind(item: Unstable_TriggerItem, isSlash: boolean): string {
  const meta = item.metadata as (RowMeta & { rawText?: string }) | undefined

  if (isSlash) {
    const group = meta?.group?.trim()

    return group === 'Skills' ? 'skill' : group === 'Themes' ? 'theme' : 'command'
  }

  // The gateway's simple refs (`@diff`, `@staged`) share one item type, so the
  // glyph comes from the directive itself.
  const raw = meta?.rawText || item.label

  if (raw.startsWith('@diff')) {
    return 'diff'
  }

  if (raw.startsWith('@staged')) {
    return 'staged'
  }

  return item.type
}

const ROW_CLASS = [
  'relative flex w-full cursor-default select-none items-center gap-2 rounded-md px-2 py-1 text-left',
  'outline-hidden transition-colors hover:bg-(--ui-bg-tertiary)',
  'data-[highlighted]:bg-(--ui-bg-tertiary) data-[highlighted]:text-foreground'
].join(' ')

const GROUP_HEADER_CLASS =
  'select-none px-2 pb-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)'

interface ComposerTriggerPopoverProps {
  activeIndex: number
  items: readonly Unstable_TriggerItem[]
  kind: '@' | '/' | ':'
  loading: boolean
  onHover: (index: number) => void
  onPick: (item: Unstable_TriggerItem) => void
  placement?: 'bottom' | 'top'
  /** The `@kind:` browse the list is filtered to, when there is one. Rendered
   *  as a header so the scope reads as the mode it is — the raw `@folder:` in
   *  the editor otherwise looks like syntax the user has to finish by hand. */
  scope?: DirectiveScope
}

/**
 * The composer's completion list, for every trigger.
 *
 * `@` and `/` render through the SAME row: icon, name, description. They used
 * to be two layouts in one file — `@` horizontal with an icon, `/` stacked with
 * none — which is why picking a file and picking a skill felt like features
 * from different apps. Icons and accents come from the shared reference
 * vocabulary, so a row looks like the chip it will become.
 *
 * `:` emoji is the one exception: the emoji IS the icon, so it renders as a
 * single display string (Slack's exact shape).
 */
export function ComposerTriggerPopover({
  activeIndex,
  items,
  kind,
  loading,
  onHover,
  onPick,
  placement = 'top',
  scope
}: ComposerTriggerPopoverProps) {
  const { t } = useI18n()
  const copy = t.composer
  const isSlash = kind === '/'
  const isEmoji = kind === ':'
  const listRef = useRef<HTMLDivElement>(null)
  // Index of the row the mouse last highlighted. Hover-driven highlights must
  // not scroll (the row is already under the cursor, and a nudge would move
  // rows under the pointer and re-fire hover in a loop) — only keyboard
  // navigation may drive the drawer's scroll position.
  const hoverIndexRef = useRef(-1)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    // The marker is single-use: it suppresses exactly the activeIndex echo of
    // the hover that set it. Clearing it on every change keeps a stale marker
    // from swallowing a later keyboard step that lands on the same index
    // (e.g. hover the last row, wrap down to 0, wrap back up).
    const isHoverEcho = activeIndex === hoverIndexRef.current

    hoverIndexRef.current = -1

    if (isHoverEcho) {
      return
    }

    const list = listRef.current

    if (!list) {
      return
    }

    // At the top, pin the drawer to 0 so the first group header stays visible;
    // `nearest` alone would stop at the row and leave the header clipped.
    if (activeIndex <= 0) {
      list.scrollTop = 0

      return
    }

    list.querySelector('[data-highlighted]')?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex, items])

  let lastGroup: string | undefined

  return (
    <div
      className={placement === 'bottom' ? COMPLETION_DRAWER_BELOW_CLASS : COMPLETION_DRAWER_CLASS}
      data-slot="composer-completion-drawer"
      data-state="open"
      onMouseDown={event => event.preventDefault()}
      onMouseLeave={() => {
        hoverIndexRef.current = -1
      }}
      ref={listRef}
      role="listbox"
    >
      {scope && <div className={cn(GROUP_HEADER_CLASS, 'pt-0.5')}>{referenceStyle(scope).label}</div>}
      {items.length === 0 ? (
        loading ? (
          <div className="flex items-center gap-2 px-2 py-1.5 text-(--ui-text-tertiary)">
            <GlyphSpinner ariaLabel={copy.lookupLoading} className="text-foreground/70" spinner="braille" />
            <span>{copy.lookupLoading}</span>
          </div>
        ) : (
          <CompletionDrawerEmpty title={copy.lookupNoMatches}>
            {kind === '@' ? (
              <>
                {copy.lookupTry} <span className="font-mono text-foreground/80">@file:</span> {copy.lookupOr}{' '}
                <span className="font-mono text-foreground/80">@folder:</span>.
              </>
            ) : isEmoji ? (
              <>
                {copy.lookupTry} <span className="font-mono text-foreground/80">:joy:</span>.
              </>
            ) : (
              <>
                {copy.lookupTry} <span className="font-mono text-foreground/80">/help</span>.
              </>
            )}
          </CompletionDrawerEmpty>
        )
      ) : (
        items.map((item, index) => {
          const meta = item.metadata as RowMeta | undefined
          const display = meta?.display ?? (isSlash ? `/${item.label}` : item.label)
          const description = meta?.meta || item.description
          const group = meta?.group?.trim()
          const showHeader = isSlash && Boolean(group) && group !== lastGroup
          const isFirstHeader = lastGroup === undefined
          lastGroup = group || lastGroup
          const active = index === activeIndex
          const refKind = referenceKind(rowKind(item, isSlash))

          return (
            <Fragment key={item.id}>
              {showHeader && <div className={cn(GROUP_HEADER_CLASS, isFirstHeader ? 'pt-0.5' : 'pt-2')}>{group}</div>}
              <button
                className={ROW_CLASS}
                data-highlighted={active ? '' : undefined}
                onClick={() => onPick(item)}
                onMouseEnter={() => {
                  hoverIndexRef.current = index
                  onHover(index)
                }}
                type="button"
              >
                {isEmoji ? (
                  // The emoji is its own icon — a glyph column beside it reads
                  // as decoration.
                  <span className="min-w-0 shrink truncate leading-5 text-foreground">{display}</span>
                ) : (
                  <>
                    <span className="grid size-4 shrink-0 place-items-center text-(--ref-color)" data-ref={refKind}>
                      <Codicon name={referenceStyle(refKind).codicon} size="0.875rem" />
                    </span>
                    <span className="min-w-0 shrink truncate font-medium leading-5 text-foreground">{display}</span>
                    {description && (
                      <span className="min-w-0 flex-1 truncate leading-5 text-(--ui-text-tertiary)">{description}</span>
                    )}
                  </>
                )}
              </button>
            </Fragment>
          )
        })
      )}
    </div>
  )
}
