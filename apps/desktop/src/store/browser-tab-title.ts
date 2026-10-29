/**
 * The document title shared by native Desktop windows and browser tabs.
 * Browser tabs also carry session status:
 *
 * In a browser the tab strip is the one surface that stays visible while the
 * user is in another tab or app, so the title carries the same signals the
 * sidebar dots and titlebar badge do, ahead of the focused session's name:
 *
 *   (2) ⚠ Fix the flaky test · Hermes
 *
 * a count of unread finished sessions, then one mark — `⚠` when anything is
 * blocked on the user, else `●` while a turn runs — then the session.
 *
 * Electron mirrors the focused caption into the native window title without
 * browser status marks. Auxiliary windows keep their own titles. Unsent drafts
 * never name the document — their text is composer content, and a title lands
 * in browser history and synced tab lists.
 */

import { computed } from 'nanostores'

import { $workspaceOwnerLabels, workspaceOwnerTitle } from '@/components/pane-shell/workspace-scope'
import { hostHas } from '@/lib/host-capabilities'

import { normalizeProfileKey } from './profile'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  ownerLookupSessionRows,
  sessionMatchesStoredId
} from './session'
import { $unreadSessionCount } from './session-dot-state'
import { $focusedStoredSessionId } from './session-focus'
import { $attentionSessionIds, $sessionTiles, $workingSessionIds } from './session-states'

const APP_NAME = 'Hermes'

export interface BrowserTabStatus {
  /** The focused session's name, or empty when it has none yet. */
  caption: string
  /** Some session in this window is blocked on an approval or an answer. */
  needsInput: boolean
  /** Listed sessions holding a finished turn the user has not opened. */
  unread: number
  /** Some session in this window has a turn running. */
  working: boolean
}

export function browserTabTitle({ caption, needsInput, unread, working }: BrowserTabStatus): string {
  // A blocking prompt is the only state that needs the user, so it speaks over
  // a running turn rather than beside it.
  const mark = needsInput ? '⚠' : working ? '●' : ''
  const name = caption ? `${caption} · ${APP_NAME}` : APP_NAME

  return [unread > 0 ? `(${unread})` : '', mark, name].filter(Boolean).join(' ')
}

const $focusedCaption = computed(
  [
    $focusedStoredSessionId,
    $sessions,
    $cronSessions,
    $messagingSessions,
    $unlistedSessionOwnerRows,
    $sessionTiles,
    $workspaceOwnerLabels
  ],
  (focused, _sessions, _cron, _messaging, _unlisted, tiles) => {
    if (!focused) {
      return ''
    }

    const row = ownerLookupSessionRows().find(session => sessionMatchesStoredId(session, focused))
    const tile = tiles.find(candidate => candidate.storedSessionId === focused)
    // Hidden relationship chats (canonical Bot Chats) are absent from the list;
    // their tab keeps a stable title, which the owner label then replaces.
    const title = row?.title?.trim() || tile?.workspaceTabTitle || ''

    if (!title) {
      return ''
    }

    const caption = workspaceOwnerTitle(title, tile)
    const profile = normalizeProfileKey(row?.profile ?? tile?.ownerProfile)

    // A bot's own name already says whose chat this is.
    return caption === title && profile !== 'default' ? `${caption} — ${profile}` : caption
  }
)

const $browserTabTitle = computed(
  [$focusedCaption, $attentionSessionIds, $workingSessionIds, $unreadSessionCount],
  (caption, attention, working, unread) =>
    browserTabTitle({ caption, needsInput: attention.length > 0, unread, working: working.length > 0 })
)

const $nativeWindowTitle = computed($focusedCaption, caption =>
  browserTabTitle({ caption, needsInput: false, unread: 0, working: false })
)

/** Keep the document title in step with the focused session. Electron mirrors
 *  it into the native window title; only browser tabs carry status marks. */
export function installBrowserTabTitle(): () => void {
  if (typeof document === 'undefined' || new URLSearchParams(window.location.search).has('win')) {
    return () => {}
  }

  const titleStore = hostHas('tabTitleStatus') ? $browserTabTitle : $nativeWindowTitle

  return titleStore.subscribe(title => {
    if (document.title !== title) {
      document.title = title
    }
  })
}
