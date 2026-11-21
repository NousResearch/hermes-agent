import type { NewChatWorkspaceTarget } from '@/store/session'

export interface FreshSessionDraftOptions {
  preserveRoute?: boolean
  replaceRoute?: boolean
  rotateFreshDraftKey?: boolean
  workspaceTarget?: NewChatWorkspaceTarget
}

export function normalizeNewChatWorkspaceTarget(target: NewChatWorkspaceTarget): NewChatWorkspaceTarget {
  return typeof target === 'string' ? target.trim() || null : target
}
