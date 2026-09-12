import { MESSAGING_SESSION_SOURCE_IDS } from '@/lib/session-source'

// The recents list is local-only: cron rows have their own section, kanban
// dispatcher workers are read on the board, and each messaging platform
// (telegram, discord, …) is fetched separately into its own self-managed
// sidebar section. Excluding them here keeps "Load more" paging through
// interactive local chats instead of interleaving gateway threads that bury them.
export const SIDEBAR_EXCLUDED_SOURCES = ['cron', 'kanban', 'subagent', 'tool', ...MESSAGING_SESSION_SOURCE_IDS]

/** Desktop-side default extras merged on top of SIDEBAR_EXCLUDED_SOURCES when
 *  `sessions.exclude_sources` is unset / invalid. An explicit empty array
 *  restores today's built-in-only list (no a2a). */
export const DEFAULT_SESSIONS_EXCLUDE_SOURCES = ['a2a']

function nonBlankStrings(values: readonly unknown[]): string[] {
  return values.filter((value): value is string => typeof value === 'string' && value.trim() !== '')
}

/** Read `sessions.exclude_sources` from a Hermes config record.
 *  Missing / null / non-array → default extras `['a2a']`.
 *  Array (including empty) → that array after dropping blank/non-string entries. */
export function extrasFromSessionsConfig(config: unknown): string[] {
  const sessions = config && typeof config === 'object' ? (config as Record<string, unknown>).sessions : undefined
  const raw =
    sessions && typeof sessions === 'object' && sessions !== null
      ? (sessions as Record<string, unknown>).exclude_sources
      : undefined

  if (!Array.isArray(raw)) {
    return [...DEFAULT_SESSIONS_EXCLUDE_SOURCES]
  }

  return nonBlankStrings(raw)
}

/** Built-in first, then extras; drop blanks/non-strings and dedupe in order.
 *  `undefined` / `null` extras use the a2a default so unset config excludes
 *  a2a without hardcoding it into SIDEBAR_EXCLUDED_SOURCES (empty extras
 *  restore today's built-in-only list). */
export function mergeSidebarRecentsExclude(builtIn: readonly string[], extras?: string[] | null): string[] {
  const extraList = extras == null ? [...DEFAULT_SESSIONS_EXCLUDE_SOURCES] : extras
  const merged: string[] = []
  const seen = new Set<string>()

  for (const source of [...builtIn, ...extraList]) {
    if (typeof source !== 'string' || source.trim() === '' || seen.has(source)) {
      continue
    }

    seen.add(source)
    merged.push(source)
  }

  return merged
}

/** Convenience: merge extras onto the built-in sidebar recents exclude list. */
export function mergedSidebarRecentsExclude(extras?: string[] | null): string[] {
  return mergeSidebarRecentsExclude(SIDEBAR_EXCLUDED_SOURCES, extras)
}

/** Fail-open config load: a thrown/rejected config fetch still excludes a2a by
 *  default so the sidebar recents/search request can complete. */
export async function resolveSidebarRecentsExclude(loadConfig: () => Promise<unknown>): Promise<string[]> {
  try {
    return mergedSidebarRecentsExclude(extrasFromSessionsConfig(await loadConfig()))
  } catch {
    return mergedSidebarRecentsExclude()
  }
}
