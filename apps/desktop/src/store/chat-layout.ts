import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.chat-layout.v1'

export type ChatLayout = 'bubbles' | 'stacked'

function normalizeChatLayout(value: unknown): ChatLayout {
  return value === 'bubbles' ? 'bubbles' : 'stacked'
}

export const $chatLayout = atom<ChatLayout>(normalizeChatLayout(storedString(KEY)))

export function setChatLayout(value: ChatLayout): void {
  $chatLayout.set(normalizeChatLayout(value))
}

if (typeof window !== 'undefined') {
  $chatLayout.subscribe(value => {
    document.documentElement.classList.toggle('chat-layout-bubbles', value === 'bubbles')
    persistString(KEY, value === 'bubbles' ? 'bubbles' : null)
  })
}
