/**
 * Canonical Bot Chat tab identity — the session titled exactly "Bot Chat"
 * must keep that caption on re-bind/restore, never a message-derived preview.
 *
 * Identity is the NAME (see repo AGENTS.md Bot Mode). `root_title` is the
 * durable lineage-root title exact-lookup gateways report; `title` covers
 * windowed listings and the stored row.
 */

export const CANONICAL_BOT_CHAT_TITLE = 'Bot Chat'

export function isCanonicalBotChatTitle(title?: null | string, rootTitle?: null | string): boolean {
  const root = String(rootTitle || '').trim()
  const stored = String(title || '').trim()

  return root === CANONICAL_BOT_CHAT_TITLE || stored === CANONICAL_BOT_CHAT_TITLE
}

/** Caption for a session tab / tile. Canonical Bot Chat always wins over a
 *  preview-derived listing title. Non-canonical sessions keep title → preview
 *  → fallback, matching `sessionTitle`. */
export function canonicalSessionTabCaption(input: {
  preview?: null | string
  rootTitle?: null | string
  title?: null | string
  untitledFallback?: string
  workspaceTabTitle?: null | string
}): string {
  if (
    isCanonicalBotChatTitle(input.title, input.rootTitle) ||
    String(input.workspaceTabTitle || '').trim() === CANONICAL_BOT_CHAT_TITLE
  ) {
    return CANONICAL_BOT_CHAT_TITLE
  }

  const title = String(input.title || '').trim()

  if (title) {
    return title
  }

  const explicit = String(input.workspaceTabTitle || '').trim()

  if (explicit) {
    return explicit
  }

  const preview = String(input.preview || '').trim()

  if (preview) {
    return preview
  }

  return input.untitledFallback ?? 'Untitled session'
}
