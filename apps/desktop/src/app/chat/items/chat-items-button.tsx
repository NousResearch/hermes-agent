import { useStore } from '@nanostores/react'

import { $paneVisible } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { $chatItemsOpen, CHAT_ITEMS_PANE_ID } from '@/store/chat-items'
import { $showsAdvancedChrome } from '@/store/interface-mode'

import type { ChatOwner } from '../use-chat-owner'

import { useChatItemsCopy } from './chat-items-copy'
import { revealChatItems } from './chat-items-pane-id'
import { useChatItems } from './use-chat-items'

/** Simple mode's way to the chat's items (1095:9309 "6 items"): the panel opens on click. Developer
 *  mode reaches the same pane from the tool rail, so the button stands down there. */
export function ChatItemsButton({ chat, sessionId }: { chat: ChatOwner; sessionId: null | string }) {
  const copy = useChatItemsCopy()
  const advanced = useStore($showsAdvancedChrome)
  const wanted = useStore($chatItemsOpen)
  const shown = useStore($paneVisible(CHAT_ITEMS_PANE_ID))
  const { total } = useChatItems(sessionId, chat)

  if (advanced || total === 0) {
    return null
  }

  return (
    <Button aria-pressed={wanted && shown} className="shrink-0" onClick={revealChatItems} size="sm" variant="ghost">
      <PixelIcon name="grid-2x2-2" />
      {copy.itemsButton(total)}
    </Button>
  )
}
