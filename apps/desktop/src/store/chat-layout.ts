import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

export type ChatLayout = 'stacked' | 'bubbles'

const CHAT_LAYOUT_STORAGE_KEY = 'hermes.desktop.chatLayout'

function loadChatLayout(): ChatLayout {
  const raw = storedString(CHAT_LAYOUT_STORAGE_KEY)
  return raw === 'bubbles' ? 'bubbles' : 'stacked'
}

export const $chatLayout = atom<ChatLayout>(loadChatLayout())

$chatLayout.subscribe(layout => persistString(CHAT_LAYOUT_STORAGE_KEY, layout))

export function setChatLayout(layout: ChatLayout) {
  $chatLayout.set(layout)
}
