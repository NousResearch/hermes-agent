import { atom } from 'nanostores'

const KEY = 'hermes.desktop.chatLayout.v1'

export type ChatLayout = 'stacked' | 'bubbles'

function loadChatLayout(): ChatLayout {
  if (typeof window === 'undefined') {
    return 'stacked'
  }

  try {
    return window.localStorage.getItem(KEY) === 'bubbles' ? 'bubbles' : 'stacked'
  } catch {
    return 'stacked'
  }
}

export const $chatLayout = atom<ChatLayout>(loadChatLayout())

$chatLayout.subscribe(layout => {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(KEY, layout)
  } catch {
    // This is a local presentation preference; ignore storage failures.
  }
})

export function setChatLayout(layout: ChatLayout) {
  $chatLayout.set(layout)
}
