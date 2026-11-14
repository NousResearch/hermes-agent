import { Codecs, persistentAtom } from '@/lib/persisted'

import { modeBound } from './interface-mode'

/** The chat items pane's id in the layout tree. */
export const CHAT_ITEMS_PANE_ID = 'chat-items'

/** The chat items pane (a chat's artifacts, files and links): remembered open in developer mode,
 *  resting closed in simple mode, where opening it is a reveal for this session (interface-mode). */
const $chatItemsOpenPref = persistentAtom('hermes.desktop.chatItemsOpen', false, Codecs.bool)

export const $chatItemsOpen = modeBound('chatItemsOpen', $chatItemsOpenPref, open => $chatItemsOpenPref.set(open))

export const openChatItems = () => $chatItemsOpen.set(true)

export const closeChatItems = () => $chatItemsOpen.set(false)
