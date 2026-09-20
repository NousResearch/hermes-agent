// The IDE chat column's active-tab pointer. Tabs themselves are this window's
// session tiles (persisted under the IDE-owned tiles key — see
// store/session-states.ts); this atom only records which one is showing, so a
// reload restores the same conversation.

import { type Codec, persistentAtom } from '@/lib/persisted'

export const IDE_CHAT_STORAGE_KEY = 'hermes.desktop.ideChat.v1'

const codec: Codec<null | string> = {
  decode: raw => (raw && raw !== 'null' ? raw : null),
  encode: value => (value === null ? null : value)
}

export const $ideActiveChat = persistentAtom<null | string>(IDE_CHAT_STORAGE_KEY, null, codec)

export function activateIdeChat(storedSessionId: null | string) {
  $ideActiveChat.set(storedSessionId)
}
