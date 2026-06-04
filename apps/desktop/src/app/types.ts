import type * as React from 'react'

import type { ChatMessage } from '@/lib/chat-messages'

export interface ContextSuggestion {
  text: string
  display: string
  meta?: string
}

export interface ImageAttachResponse {
  attached?: boolean
  path?: string
  text?: string
  message?: string
}

export interface ImageDetachResponse {
  detached?: boolean
  count?: number
}

export interface SlashExecResponse {
  output?: string
  warning?: string
}

export interface ExecCommandDispatchResponse {
  type: 'exec' | 'plugin'
  output?: string
}

export interface AliasCommandDispatchResponse {
  type: 'alias'
  target: string
}

export interface SkillCommandDispatchResponse {
  type: 'skill'
  name: string
  message?: string
}

export interface SendCommandDispatchResponse {
  type: 'send'
  message: string
}

export type CommandDispatchResponse =
  | ExecCommandDispatchResponse
  | AliasCommandDispatchResponse
  | SkillCommandDispatchResponse
  | SendCommandDispatchResponse

export type SidebarNavId = 'artifacts' | 'command-center' | 'messaging' | 'new-session' | 'settings' | 'skills'

export interface SidebarNavItem {
  id: SidebarNavId
  label: string
  icon: React.ComponentType<{ className?: string }>
  route?: string
  action?: 'new-session'
}

export interface ClientSessionState {
  storedSessionId: string | null
  messages: ChatMessage[]
  branch: string
  cwd: string
  busy: boolean
  awaitingResponse: boolean
  streamId: string | null
  sawAssistantPayload: boolean
  pendingBranchGroup: string | null
  interrupted: boolean
}
