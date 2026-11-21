import { useI18n } from '@/i18n'
import { NEW_SESSION_TITLE } from '@/lib/chat-runtime'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $draftTitles, draftTitleIn } from '@/store/composer'
import { usePortalShell } from '@/store/shell-variant'

export interface SessionDraftTitleProps {
  /** The draft's composer key — a tile's stored session id, or null for the
   *  new chat that has no session yet. */
  scope: null | string
}

/**
 * A DRAFT'S NAME — what an unsent session is called until it has a real one.
 *
 * The tab of a session that has never been sent renders this instead of its
 * registered title, because the name moves with the composer: every debounced
 * stash republishes it. Re-registering the contribution at that rate would
 * re-render the whole panes area, so the label subscribes for itself and its
 * own key only.
 *
 * Falls back to the placeholder rather than going blank, so an emptied composer
 * reads the same as one never typed into.
 */
export function SessionDraftTitle({ scope }: SessionDraftTitleProps) {
  const { t } = useI18n()
  const portal = usePortalShell()
  const draft = useStoreSelector($draftTitles, titles => draftTitleIn(titles, scope))

  // Classic keeps main's fixed English placeholder; the redesign names it in its own wording.
  return draft || (portal ? t.sidebar.nav['new-session'] : NEW_SESSION_TITLE)
}
