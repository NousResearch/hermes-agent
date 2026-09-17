import { session } from 'electron'

import { CHATGPT_SESSION_PARTITION, configureChatGptSession } from './chatgpt-session'

export function installChatGptSession(): void {
  configureChatGptSession(session.fromPartition(CHATGPT_SESSION_PARTITION))
}