/**
 * Require ⌘-click (Mac) or Ctrl-click (elsewhere) to open chat links: a
 * device-local preference, off by default.
 *
 * On: a plain mouse click on a link inside assistant prose (an authored URL,
 * a bare path, a path or URL code chip) selects text instead of opening it,
 * and the platform modifier opens it in-app. A keyboard activation (Enter on
 * a focused link, `detail === 0`) still opens. Links outside chat prose
 * (settings, plugins, artifacts) never consult this.
 *
 * Same shape as `external-links.ts`: persisted per machine, and the `storage`
 * listener keeps every open window in step.
 */

import { atom } from 'nanostores'

import { IS_MAC } from '@/lib/keybinds/combo'
import { persistStore } from '@/lib/persist-store'
import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.chatLinksRequireModifier.v1'

/** Set on `<html>` while the preference is on, so CSS can drop the pointer
 *  cursor from gated chat links without a per-link subscription. */
export const CHAT_LINKS_REQUIRE_MODIFIER_ATTR = 'data-chat-links-require-modifier'

/** Marks a link that honours the preference. */
export const CHAT_LINK_ATTR = 'data-chat-link'

export const $chatLinksRequireModifier = atom<boolean>(
  typeof window === 'undefined' ? false : storedBoolean(KEY, false)
)

export function setChatLinksRequireModifier(on: boolean): void {
  $chatLinksRequireModifier.set(on)
}

/** Whether this click on a chat link should open it. */
export function shouldOpenChatLink(event: { ctrlKey: boolean; detail: number; metaKey: boolean }): boolean {
  if (!$chatLinksRequireModifier.get() || event.detail === 0) {
    return true
  }

  return IS_MAC ? event.metaKey : event.ctrlKey
}

if (typeof window !== 'undefined') {
  persistStore($chatLinksRequireModifier, (on, options) => persistBoolean(KEY, on, options), false)

  $chatLinksRequireModifier.subscribe(on => {
    document.documentElement.toggleAttribute(CHAT_LINKS_REQUIRE_MODIFIER_ATTR, on)
  })

  window.addEventListener('storage', event => {
    if (event.key === KEY) {
      $chatLinksRequireModifier.set(storedBoolean(KEY, false))
    }
  })
}
