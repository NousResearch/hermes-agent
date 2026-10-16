import { $temporaryDraft, registerTemporaryChat } from '@/store/temporary-chat'

/**
 * `session.create` params for a temporary draft. Deliberately NOT droppable (see
 * DROPPABLE_CREATE_FIELDS): a backend that predates temporary chats must refuse the create, not
 * silently save the conversation the user asked to keep out of history.
 */
export function temporaryCreateParams(): { temporary?: true } {
  return $temporaryDraft.get() ? { temporary: true } : {}
}

/**
 * Record a just-created temporary session (stored + runtime id) instead of seeding a sidebar
 * row: the backend never lists it. True when the create was temporary.
 */
export function adoptTemporarySession(
  params: Record<string, unknown>,
  runtimeSessionId: string,
  storedSessionId: string
): boolean {
  if (params.temporary !== true) {
    return false
  }

  registerTemporaryChat(storedSessionId, runtimeSessionId)

  return true
}

/** A new chat's sidebar seed: a temporary chat is registered instead (never listed). */
export function seedCreatedSession(
  params: Record<string, unknown>,
  runtimeSessionId: string,
  storedSessionId: string,
  seedSidebarRow: () => void
): void {
  if (!adoptTemporarySession(params, runtimeSessionId, storedSessionId)) {
    seedSidebarRow()
  }
}
