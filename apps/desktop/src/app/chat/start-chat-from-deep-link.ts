/**
 * `hermes://chat/new?prompt=…` — open a fresh chat and seed its composer.
 *
 * The website's "Run in Hermes" buttons point here: a visitor reading a page
 * that tells them to run something gets that text waiting in a new chat
 * instead of retyping it.
 *
 * It seeds and never sends. The text lands in the composer and stops there, so
 * the user reads it and presses Enter themselves: a web page must never be able
 * to start a turn on the user's machine.
 *
 * It opens the chat the way Cmd+N does: `requestFreshSession()` is the request
 * atom the Cmd+W close path and the resume-exhausted overlay publish, and the
 * workspace/profile reset mirrors the `session.new` keybind, so a link lands in
 * the Sessions workspace on the live profile rather than inside whatever Bot
 * Mode zone was open.
 *
 * The prompt rides the fresh-draft rotation (`seedNextFreshDraft`), not the
 * composer insert bus. The bus fires on a macrotask that races the render that
 * swaps the composer to the new chat's draft scope; when it wins, the text
 * paints into the chat the user was looking at and is stashed under that
 * chat's key. The seed is written under the new key before the composer can
 * see it, so the scope swap loads it like any restored draft.
 */

import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { seedNextFreshDraft } from '@/store/composer'
import { $newChatProfile, requestFreshSession } from '@/store/profile'

import { requestComposerFocus } from './composer/focus'

export function startChatFromDeepLink(prompt: string): void {
  setWorkspaceScope('sessions')
  $newChatProfile.set(null)
  seedNextFreshDraft(prompt)
  requestFreshSession()
  requestComposerFocus('main')
}
