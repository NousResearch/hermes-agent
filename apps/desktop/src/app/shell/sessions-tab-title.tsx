import { translateNow } from '@/i18n'
import { compactNumber } from '@/lib/format'

interface SessionsTabTitleProps {
  /** Live unread count — subscribed by the caller, so this stays a pure
   *  presentational component that render tests can drive directly. */
  unread: number
  onOpenNextUnread: () => void
}

/** An unread count is navigation, not layout state. Keep it on the Sessions
 *  tab while the sidebar is visible; the titlebar reveal control takes over
 *  only while the whole sidebar is hidden. */
export function SessionsTabTitle({ onOpenNextUnread, unread }: SessionsTabTitleProps) {
  const unreadLabel = unread > 0 ? translateNow('titlebar.unreadSessions', unread) : ''

  return (
    <span className="inline-flex min-w-0 items-center gap-1.5">
      <span>sessions</span>
      {unread > 0 ? (
        // Reserve two digit slots (tabular-nums equalizes digit widths, not
        // the node's own box) so 9 → 10 doesn't re-flow the tab label.
        <button
          aria-label={unreadLabel}
          className="inline-block min-w-[2ch] shrink-0 text-center text-(--ui-accent) tabular-nums outline-none hover:bg-(--chrome-action-hover) focus-visible:ring-1 focus-visible:ring-(--ui-accent)"
          onClick={event => {
            event.stopPropagation()
            onOpenNextUnread()
          }}
          onPointerDown={event => event.stopPropagation()}
          type="button"
        >
          {compactNumber(unread)}
        </button>
      ) : null}
    </span>
  )
}
