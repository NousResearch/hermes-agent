import { revealTreePane } from '@/components/pane-shell/tree/store'
import { CHAT_ITEMS_PANE_ID, openChatItems } from '@/store/chat-items'

import { prepareDeveloperToolPanel } from '../../redesign/developer-tool-panel'

/** Open the chat's items beside it, in the one side panel the developer tools share. The panel is
 *  placed before the reveal: a reveal un-folds the side its pane is on, and simple mode rests with
 *  the tools' side folded. */
export function revealChatItems(): void {
  openChatItems()
  prepareDeveloperToolPanel(CHAT_ITEMS_PANE_ID)
  revealTreePane(CHAT_ITEMS_PANE_ID)
}
