/**
 * Roster payload fingerprints for content-change detection.
 *
 * The desktop polls the roster every 5s (useRoster refetchInterval) and React
 * Query hands back a fresh object graph each time. The pane republishes that
 * graph to shared atoms — a new array reference re-renders every subscriber
 * and re-runs avatar/meta/label side effects even when nothing changed; on
 * macOS that churn drops IME input-method focus. These helpers reduce a
 * payload to a stable content string so the pane can skip the whole publish
 * when the roster is substantively identical.
 *
 * The fingerprint deliberately covers only what the pane's consumers observe:
 * identity, labels, source/status, session activity, avatar flags and the
 * server `hermes-bots` ui_meta overlay. Transient envelope noise (fetchedAt),
 * write bookkeeping (ui_meta_revisions) and foreign ui_meta keys are left out
 * — none of the shared-roster read paths consume them, so an unchanged
 * fingerprint means an unchanged observable roster.
 */

import type { BotMeta, CanonicalSession, RosterRow, SessionPreview } from './types'

/** Deterministic serializer: nested wire objects (route, ui_meta) can arrive
 *  with any key order, so sort keys instead of trusting insertion order. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return String(JSON.stringify(value))
  }

  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`
  }

  const object = value as Record<string, unknown>

  return `{${Object.keys(object)
    .sort()
    .map(key => `${JSON.stringify(key)}:${stableStringify(object[key])}`)
    .join(',')}}`
}

/** Normalize a canonical/last session to the fields that drive activity
 *  signals (age label, pulse dot, unread watermark, recency sort) and
 *  previews. Not in the normalized shape: nothing else about a session
 *  reaches the pane or the shared-roster consumers. */
function sessionFingerprint(session: CanonicalSession | SessionPreview | null | undefined) {
  if (!session || typeof session !== 'object') {
    return null
  }

  const { id, last_active, preview, resolved_id, root_title, title } = session as CanonicalSession

  return { id, last_active, preview, resolved_id, root_title, title }
}

/** Server `hermes-bots` ui_meta minus the local-only payload bits: image
 *  data URLs (and derived pet state) never come from the server
 *  (mergeServerMeta preserves them locally), and a multi-KB data URL must not
 *  ride the fingerprint — it would be re-serialized on every poll. */
function metaFingerprint(meta: BotMeta | null | undefined) {
  if (!meta || typeof meta !== 'object') {
    return null
  }

  const copy = { ...meta }
  delete copy.image

  return stableStringify(copy)
}

/** One stable content string for a roster payload: non-ghost rows reduced to
 *  their observable fields, ordered by (connectionId, name, content) so wire
 *  order and the pane's pin/activity sort never count as change. */
export function rosterPayloadFingerprint(roster: RosterRow[]): string {
  const entries = (Array.isArray(roster) ? roster : [])
    .filter(row => !row?.ghost)
    .map(row => {
      const meta = row.ui_meta?.['hermes-bots']

      // Fixed key order: an identical payload hashes identically.
      const normalized = {
        canonical_session: sessionFingerprint(row.canonical_session),
        connectionId: row.connectionId,
        connectionKind: row.connectionKind,
        connectionLabel: row.connectionLabel,
        description: row.description,
        display_name: row.display_name,
        handle: row.handle,
        has_avatar: row.has_avatar,
        last_session: sessionFingerprint(row.last_session),
        name: row.name,
        remoteSource: row.remoteSource,
        route: row.route ? stableStringify(row.route) : null,
        sourceError: row.sourceError,
        sourceMissing: row.sourceMissing,
        sourceReachable: row.sourceReachable,
        sourceScoped: row.sourceScoped,
        targetProfile: row.targetProfile,
        title: row.title,
        ui_meta: metaFingerprint(meta),
        worker_session: row.worker_session ? { last_active: row.worker_session.last_active } : null
      }

      const content = stableStringify(normalized)
      const key = `${row.connectionId || ''}\u0000${row.name}\u0000${content}`

      return { content, key }
    })
    .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))

  return entries.map(entry => entry.content).join('\n')
}