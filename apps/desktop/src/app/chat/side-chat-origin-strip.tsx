/**
 * The "where this side chat came from" strip.
 *
 * A side chat is an ordinary session — that is what makes it independent and
 * what lets it reuse the tile chassis — so nothing on the session itself says it
 * was opened from a selection in another conversation. Without this, the pane
 * restored a day later is a transcript whose quoted opening block refers to
 * something the user can no longer see.
 *
 * Dismissible on purpose: provenance is information, not chrome, and the user is
 * allowed to decide they no longer need it.
 */

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { dismissSideChatOriginBanner, type SideChatOrigin } from '@/store/side-chat'

export function SideChatOriginStrip({
  origin,
  storedSessionId
}: {
  origin: SideChatOrigin
  storedSessionId: string
}) {
  const { t } = useI18n()
  const copy = t.desktop.sideChat

  return (
    <div
      className="flex shrink-0 items-center justify-between gap-2 border-b border-(--stroke-nous) bg-(--ui-chat-bubble-background) px-3 py-1.5 text-xs text-muted-foreground"
      data-side-chat-origin={storedSessionId}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <Codicon name="comment-discussion" size="0.75rem" />
        <span className="truncate">
          {origin.fromTitle ? copy.referencingFrom(origin.fromTitle) : copy.chatAboutSelection}
        </span>
      </span>
      <button
        className="shrink-0 rounded px-1.5 py-0.5 hover:bg-accent hover:text-foreground"
        onClick={() => dismissSideChatOriginBanner(storedSessionId)}
        type="button"
      >
        {copy.dismiss}
      </button>
    </div>
  )
}
