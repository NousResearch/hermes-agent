import type { BrowserConversation } from '../../electron/browser-workspace-types'

import { currentSessionId } from './preview-ownership'
import { isSessionGone } from './session-gone-latch'

// The session store installs its established durable-id/owner resolver. Keep
// this leaf free of session-states imports: preview is itself its dependency.
let resolveSession: (id: string) => BrowserConversation | null = () => null
let focusedSession: () => string | null = () => null
const retired = new Set<string>()

export function setBrowserSessionResolver(resolve: typeof resolveSession, focused: typeof focusedSession = () => null) {
  resolveSession = resolve
  focusedSession = focused
}

export function focusedBrowserConversation(): BrowserConversation | null {
  return browserSessionConversation(focusedSession() || '')
}

export function browserSessionConversation(id: string): BrowserConversation | null {
  if (!id || retired.has(id) || isSessionGone(id)) {return null}
  const owner = resolveSession(id)
  const tip = currentSessionId(owner?.id ?? id)!

  return owner && !retired.has(tip) && !isSessionGone(tip) ? { ...owner, id: tip } : null
}

export function isBrowserSessionRetired(id: string): boolean {
  const owner = resolveSession(id)

  const tip = currentSessionId(owner?.id ?? id)!

  return retired.has(id) || isSessionGone(id) || retired.has(tip) || isSessionGone(tip)
}

export function retireBrowserSession(id: string) {
  const tip = currentSessionId(resolveSession(id)?.id ?? id)!
  retired.add(id)
  retired.add(tip)
  window.hermesDesktop?.browserWorkspace?.retireSession?.(tip)
}

export function browserRequestSourceMismatch(source: { connectionId?: string; profile?: string }, sessionId: string): boolean {
  const owner = browserSessionConversation(sessionId)

  return Boolean(owner && (
    (source.connectionId || 'local') !== owner.connectionId ||
    (source.profile && source.profile !== owner.profile)
  ))
}

export function browserRequestConversation(source: { connectionId?: string; profile?: string }, sessionId: string): BrowserConversation | null {
  return browserRequestSourceMismatch(source, sessionId) ? null : browserSessionConversation(sessionId)
}
