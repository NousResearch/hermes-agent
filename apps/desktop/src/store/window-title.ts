/**
 * The native window title of a popped-out session window.
 *
 * A secondary session window (`?win=secondary`) shows exactly one chat, so the
 * taskbar has no other way to name it: every pop-out used to read `Hermes`,
 * which is a row of identical buttons in Alt-Tab. This titles the OS window
 * after the conversation it holds, and follows that title live — the gateway
 * pushes `session.title` when the async titler writes it, and a rename from any
 * window patches the same store.
 *
 * The mark is the same one the browser tab carries (`⚠` blocked, else `●` while
 * a turn runs), but scoped to THIS window's session: the underlying status sets
 * are profile-wide, and a pop-out of chat A must not read `⚠` because chat B is
 * blocked. The unread count is left out — it describes the profile, not this
 * window.
 *
 * Native + secondary only. A browser-hosted pop-out is an ordinary tab, which
 * `browser-tab-title.ts` already titles with the full status form.
 */

import { computed } from 'nanostores'

import { hostHas } from '@/lib/host-capabilities'

import { $focusedSessionCaption } from './browser-tab-title'
import { $focusedStoredSessionId } from './session-focus'
import { $attentionSessionIds, $workingSessionIds } from './session-states'
import { isSecondaryWindow } from './windows'

const APP_NAME = 'Hermes'

export interface WindowTitleStatus {
  /** The window's session title, or empty when it has none yet. */
  caption: string
  /** THIS window's session is blocked on an approval or an answer. */
  needsInput: boolean
  /** THIS window's session has a turn running. */
  working: boolean
}

export function windowTitle({ caption, needsInput, working }: WindowTitleStatus): string {
  // A blocking prompt needs the user, so it speaks over a running turn rather
  // than beside it — same precedence as the browser tab title.
  const mark = needsInput ? '⚠' : working ? '●' : ''
  const name = caption || APP_NAME

  return mark ? `${mark} ${name}` : name
}

// `$attentionSessionIds` / `$workingSessionIds` are published under every id
// the conversation answers to (lineage aliases), so an exact id test survives a
// compression tip rotation.
const $ownStatus = computed(
  [$focusedStoredSessionId, $attentionSessionIds, $workingSessionIds],
  (focused, attention, working) => ({
    needsInput: focused !== null && attention.includes(focused),
    working: focused !== null && working.includes(focused)
  })
)

/** Keep the OS window title on the session this window shows. A no-op outside
 *  an Electron session window; returns the unsubscribe. */
export function installWindowTitle(): () => void {
  if (typeof document === 'undefined' || !hostHas('windowTitleStatus') || !isSecondaryWindow()) {
    return () => {}
  }

  const $shown = computed([$focusedSessionCaption, $ownStatus], (caption, status) =>
    windowTitle({ caption, ...status })
  )

  // Electron mirrors `document.title` into the native window title unless a
  // `page-title-updated` listener cancels it; nothing in electron/ does.
  return $shown.subscribe(title => {
    if (document.title !== title) {
      document.title = title
    }
  })
}
