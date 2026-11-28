import { adoptDraftToolPanel } from '@/app/redesign/chat-tool-panel'
import { announceNewSessionDraftKey } from '@/store/composer'
import { adoptDraftPreviewTabs } from '@/store/preview'

/** A new chat takes over what its pre-session draft gathered, once the create returns its stored id. */
export function adoptNewSessionDraft(stored: string): void {
  // Anything still parked under the pre-session draft bucket belongs to this chat now (#114122); the
  // composer moves it on scope swap.
  announceNewSessionDraftKey(stored)
  // The draft's preview tabs follow it the same way (#73890).
  adoptDraftPreviewTabs(stored)
  // ...and so does the tool panel it opened (redesign).
  adoptDraftToolPanel(stored)
}
