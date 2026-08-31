/**
 * THE TIP BUBBLE — a pointer, not an overlay.
 *
 * The app's popover, in its accent variant: same box, same arrow, same
 * placement engine, filled instead of glassed. Nothing here re-implements the
 * surface — a tip that drifted from the popover's shape would read as a
 * different app talking.
 *
 * What it does own is behaviour, and a coachmark's is the opposite of a menu's
 * at every point: it never takes focus, clicking the app does not dismiss it,
 * and it does not own Esc. The composer already answers Esc, and a dismissable
 * layer in front of it would make one cancel gesture do two things (DESIGN.md).
 * A tip blocks nothing and closes itself, so it never needs to compete.
 */

import { useEffect, useRef } from 'react'

import { KbdCombo } from '@/components/ui/kbd'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { iconSize, X } from '@/lib/icons'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import type { TipSide } from '@/lib/tips/catalog'

export interface TipBubbleProps {
  /** The element the arrow points at. */
  anchor: HTMLElement
  /** Keybind action id; its live combo prints under the text. */
  keybind?: string
  /** Hard close — this tip never comes back. */
  onClose: () => void
  side: TipSide
  text: string
  title?: string
}

export function TipBubble({ anchor, keybind, onClose, side, text, title }: TipBubbleProps) {
  const { t } = useI18n()
  const combo = useKeybindHint(keybind ?? '')
  const anchorRef = useRef<HTMLElement | null>(anchor)

  // Radix reads `virtualRef.current` on every render of the anchor, so keeping
  // the ref current is what lets a tip follow an element that got re-created
  // (a re-render swaps the node, not the selector).
  anchorRef.current = anchor

  // Nothing in here is a focus trap, but a tip arriving under an open native
  // menu would still float above it. Scroll it into agreement instead of
  // guessing: the popover re-measures whenever its anchor moves.
  useEffect(() => {
    anchor.scrollIntoView?.({ block: 'nearest', inline: 'nearest' })
  }, [anchor])

  return (
    <Popover open>
      <PopoverAnchor virtualRef={anchorRef} />
      <PopoverContent
        aria-live="polite"
        className="p-2.5"
        collisionPadding={12}
        data-slot="tip-bubble"
        // Ambient chrome: it must not take the caret out of the composer, and
        // touching the app is not a dismissal gesture — the ✕ and the timer
        // are. See the header note on Esc.
        onCloseAutoFocus={event => event.preventDefault()}
        onEscapeKeyDown={event => event.preventDefault()}
        onFocusOutside={event => event.preventDefault()}
        onInteractOutside={event => event.preventDefault()}
        onOpenAutoFocus={event => event.preventDefault()}
        role="status"
        side={side}
        variant="accent"
      >
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            {title && <p className="text-[length:var(--conversation-caption-font-size)] font-semibold">{title}</p>}
            {/* Held off full strength so the title still leads. Everything here
                is currentColor-relative, so it follows whatever the accent's
                foreground is rather than pinning a grey that only works on glass. */}
            <p className="mt-0.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-current/85">
              {text}
            </p>
            {combo && <KbdCombo className="mt-2" combo={combo} size="sm" variant="inverted" />}
          </div>
          <button
            aria-label={t.tips.close}
            className="-mr-0.5 -mt-0.5 shrink-0 cursor-pointer rounded-[3px] p-0.5 text-current/70 transition-colors hover:bg-current/15 hover:text-current"
            onClick={onClose}
            type="button"
          >
            <X className={iconSize.sm} />
          </button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
