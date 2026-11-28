import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { Fragment, type RefObject, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { referenceKind, referenceStyle } from '@/components/assistant-ui/reference-kinds'
import { composerPanelCard } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { CompletionDrawerEmpty } from './completion-drawer'
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
  'focus-visible:outline-hidden data-[highlighted]:outline-hidden transition-colors hover:bg-(--ui-bg-tertiary)',
  'data-[highlighted]:bg-(--ui-bg-tertiary) data-[highlighted]:text-foreground'
].join(' ')

const GROUP_HEADER_CLASS =
  'select-none px-2 pb-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)'

// The drawer portals to the body for the same reason Radix menus do: a window-top
// tree group clips its zone (`overflow: clip`), and on the centered landing screen
// or a short split tile that clip cut the `/`, `@` and `:` list off at the top.
// The skin is shared with the attach menu; only position and the viewport cap are
// this component's.
const PORTAL_DRAWER_CLASS = cn(
  'fixed z-(--z-modal-popover) w-80 max-w-[calc(100vw-1rem)] overflow-y-auto overscroll-contain p-1 text-popover-foreground',
  composerPanelCard
)

type DrawerPlacement = 'bottom' | 'top'

interface DrawerBox {
  left: number
  maxHeight: number
  top: number
  width: number
}

function remToPx(value: number): number {
  const rootSize = Number.parseFloat(getComputedStyle(document.documentElement).fontSize)

  return (Number.isFinite(rootSize) ? rootSize : 16) * value
}

/** The box `absolute bottom-full` used to resolve against (its offset parent). */
function anchorBlock(anchor: HTMLElement): HTMLElement {
  const parent = anchor.offsetParent

  return parent instanceof HTMLElement ? parent : (anchor.parentElement ?? document.body)
}

/**
 * The shell root: the outermost node under `<body>` that holds this drawer's
 * anchor. Portalling there — rather than `<body>` — keeps the drawer inside the
 * app's own theme/zoom frame (browser-hosted windows scale `#root`) while still
 * stepping outside the window-top zones that clip it. In a bare document (a test
 * container) that is the container itself; inside the HUD window it is the HUD
 * shell, whose background, dimming, backdrop and frost the drawer must sit on.
 */
function shellPortalHost(anchor: HTMLElement): HTMLElement {
  // A HUD window's shell is the window: escaping it would paint into the hidden main one.
  const hud = anchor.closest<HTMLElement>('[data-hud-shell]')

  if (hud) {
    return hud
  }

  let host = anchor.parentElement ?? document.body

  while (host.parentElement && host.parentElement !== document.body) {
    host = host.parentElement
  }

  return host
}

function sameBox(left: DrawerBox | null, right: DrawerBox): boolean {
  return Boolean(
    left &&
      left.left === right.left &&
      left.maxHeight === right.maxHeight &&
      left.top === right.top &&
      left.width === right.width
  )
}

/**
 * Viewport-anchored placement for the portalled drawer: above the anchor like the
 * old absolute card, flipped below when there is no room above, and capped to the
 * space actually left on screen so a short tile scrolls the list instead of
 * painting it off-window.
 */
function useAnchoredDrawer(
  placement: DrawerPlacement,
  anchorRef: RefObject<HTMLSpanElement | null>,
  drawerRef: RefObject<HTMLDivElement | null>
): DrawerBox | null {
  const [box, setBox] = useState<DrawerBox | null>(null)

  const measure = useCallback(() => {
    const anchor = anchorRef.current
    const drawer = drawerRef.current

    if (!anchor || !drawer) {
      return
    }

    const block = anchorBlock(anchor)
    const blockRect = block.getBoundingClientRect()
    const gap = remToPx(0.25)
    const edge = remToPx(0.5)
    const viewportWidth = document.documentElement.clientWidth
    const viewportHeight = document.documentElement.clientHeight
    const width = Math.max(0, Math.min(remToPx(20), viewportWidth - edge * 2))
    const contentHeight = drawer.scrollHeight
    const maxDrawerHeight = remToPx(22)
    const spaceAbove = blockRect.top - gap - edge
    const spaceBelow = viewportHeight - blockRect.bottom - gap - edge
    const wanted = Math.min(contentHeight, maxDrawerHeight)
    const openBelow = placement === 'bottom' || (spaceAbove < wanted && spaceBelow > spaceAbove)
    const maxHeight = Math.max(0, Math.min(maxDrawerHeight, openBelow ? spaceBelow : spaceAbove))
    const top = openBelow
      ? Math.min(blockRect.bottom + gap, viewportHeight - edge)
      : Math.max(edge, blockRect.top - gap - Math.min(contentHeight, maxHeight))
    const left = Math.min(
      Math.max(edge, blockRect.left + block.clientLeft + remToPx(0.5)),
      viewportWidth - width - edge
    )

    setBox(previous => {
      const next = { left, maxHeight, top, width }

      return sameBox(previous, next) ? previous : next
    })
  }, [anchorRef, drawerRef, placement])

  // Every render may change the row set (a keystroke filters the list), so the
  // height is re-measured after commit rather than on a dependency list.
  useLayoutEffect(measure)

  useEffect(() => {
    window.addEventListener('resize', measure)
    // Capture catches the composer's own scroller and the floating window's, not
    // just the page's. Repositioning while open mirrors what an anchored menu does.
    window.addEventListener('scroll', measure, true)

    return () => {
      window.removeEventListener('resize', measure)
      window.removeEventListener('scroll', measure, true)
    }
  }, [measure])

  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') {
      return undefined
    }

    const drawer = drawerRef.current

    if (!drawer) {
      return undefined
    }

    const observer = new ResizeObserver(measure)
    observer.observe(drawer)

    return () => observer.disconnect()
  }, [drawerRef, measure])

  return box
}

interface ComposerTriggerPopoverProps {
  activeIndex: number
  items: readonly Unstable_TriggerItem[]
  kind: '@' | '/' | ':'
  loading: boolean
  onHover: (index: number) => void
  onPick: (item: Unstable_TriggerItem) => void
  placement?: DrawerPlacement
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
  const anchorRef = useRef<HTMLSpanElement>(null)
  const hoverIndexRef = useRef(-1)
  const [portalHost, setPortalHost] = useState<HTMLElement | null>(null)
  const box = useAnchoredDrawer(placement, anchorRef, listRef)

  useLayoutEffect(() => {
    const anchor = anchorRef.current
    const host = anchor ? shellPortalHost(anchor) : null

    setPortalHost(previous => (previous === host ? previous : host))
  })

  // Only keyboard navigation should move the drawer. A hover echo already points
  // at a visible row and scrolling it can shift another row under the pointer.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const list = listRef.current

    if (!list) {
      return
    }

    const isHoverEcho = activeIndex === hoverIndexRef.current

    hoverIndexRef.current = -1

    if (isHoverEcho) {
      return
    }

    if (activeIndex === 0) {
      // `nearest` keeps the first row visible but can leave its group header
      // clipped, so wrapping to the beginning restores the complete top edge.
      list.scrollTop = 0

      return
    }

    const highlighted = list.querySelector<HTMLElement>('[data-highlighted]')

    if (!highlighted) {
      return
    }

    // Keep scrolling local to the drawer. `scrollIntoView` may also move the
    // transcript or window because it operates on every scrollable ancestor.
    const listRect = list.getBoundingClientRect()
    const highlightedRect = highlighted.getBoundingClientRect()
    const visibleTop = listRect.top + list.clientTop
    const visibleBottom = visibleTop + list.clientHeight
    const topDelta = highlightedRect.top - visibleTop
    const bottomDelta = highlightedRect.bottom - visibleBottom
    const overflowsTop = topDelta < 0
    const overflowsBottom = bottomDelta > 0

    // A row that is fully visible needs no movement. An oversized row that
    // spans both edges already covers the viewport, so moving it would not
    // reveal the whole row and would only add churn. Otherwise align whichever
    // edge requires the shorter movement, matching `block: nearest` semantics.
    if (overflowsTop === overflowsBottom) {
      return
    }

    list.scrollTop += Math.abs(topDelta) < Math.abs(bottomDelta) ? topDelta : bottomDelta
  }, [activeIndex, items, portalHost])

  let lastGroup: string | undefined

  const drawer = (
    <div
      className={PORTAL_DRAWER_CLASS}
      data-slot="composer-completion-drawer"
      data-state="open"
      onMouseDown={event => event.preventDefault()}
      ref={listRef}
      role="listbox"
      style={
        box ? { left: box.left, maxHeight: box.maxHeight, top: box.top, width: box.width } : { visibility: 'hidden' }
      }
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
              <Tip delayDuration={400} label={kind === '/' ? description : undefined} placement="row" sideOffset={4}>
                <button
                  className={ROW_CLASS}
                  data-highlighted={active ? '' : undefined}
                  onClick={() => onPick(item)}
                  onMouseEnter={() => {
                    // React bails out when hovering the already-active row. Do
                    // not leave a marker behind for a later items refresh.
                    hoverIndexRef.current = index === activeIndex ? -1 : index
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
                        <span className="min-w-0 flex-1 truncate leading-5 text-(--ui-text-tertiary)">
                          {description}
                        </span>
                      )}
                    </>
                  )}
                </button>
              </Tip>
            </Fragment>
          )
        })
      )}
    </div>
  )

  return (
    <>
      {/* The anchor keeps the drawer's old containing block measurable without
          adding layout of its own. */}
      <span aria-hidden className="pointer-events-none absolute left-0 top-0 h-0 w-0" ref={anchorRef} />
      {portalHost && createPortal(drawer, portalHost)}
    </>
  )
}
