import { setInterfaceMode } from '@/store/interface-mode'

const CHAT_LAYOUT_KEY = 'hermes.desktop.ivxChatLayout.v1'

/** Opens the chat layout the first time this build runs. Switching back to
 *  Advanced afterwards stays, because the flag is written once. */
export function applyChatLayoutOnce() {
  try {
    if (window.localStorage.getItem(CHAT_LAYOUT_KEY) === '1') {
      return
    }

    window.localStorage.setItem(CHAT_LAYOUT_KEY, '1')
    setInterfaceMode('simple')
  } catch {
    // Storage can be missing. The current layout remains.
  }
}
