import type { GroupChat, GroupMessage } from './types'

/** A token is the title-edit generation at the start of one async read-back.
 * Track only keys edited while it is in flight, across every gateway. Tokens
 * retire in finally, so this state cannot accumulate across completed jobs. */
export function createThreadTitleReadFence() {
  const active = new Set<Map<string, Set<string>>>()

  return {
    begin() {
      const token = new Map<string, Set<string>>()

      active.add(token)

      return token
    },
    end(token: Map<string, Set<string>>) { active.delete(token) },
    record(group: string, ids: string[]) {
      if (!ids.length) { return }

      for (const token of active) {
        token.set(group, new Set([...(token.get(group) || []), ...ids]))
      }
    },
    protect(token: Map<string, Set<string>>, current: Record<string, GroupChat>, merged: Record<string, GroupChat>) {
      for (const [group, ids] of token) {
        if (!current[group] || !merged[group] || !ids.size) { continue }
        const titles = current[group].threadTitles || {}

        merged[group] = { ...merged[group], threadTitles: {
          ...merged[group].threadTitles,
          ...Object.fromEntries([...ids].flatMap(id => Object.prototype.hasOwnProperty.call(titles, id) ? [[id, titles[id]]] : []))
        } }
      }

      return merged
    }
  }
}

export const GROUP_THREAD_TITLE_CHARS = 120

export function boundedThreadTitle(title: string): string {
  // Unicode-mode matching replaces only lone surrogates, never an astral pair.
  return Array.from(title.trim()).slice(0, GROUP_THREAD_TITLE_CHARS).join('').replace(/[\uD800-\uDFFF]/gu, '\uFFFD')
}

/** Publish intent is queued per room/thread, not inferred from a new room
 * revision. Union keys across debounce, retry and gateway fan-out; values come
 * from the latest local snapshot so a retry never reinstates an older edit. */
export function mergeThreadTitleEdits(
  earlier: Record<string, string[]> = {},
  later: Record<string, string[]> = {}
): Record<string, string[]> {
  return Object.fromEntries([...new Set([...Object.keys(earlier), ...Object.keys(later)])].map(key => [
    key, [...new Set([...(earlier[key] || []), ...(later[key] || [])])]
  ]))
}

/** Carry only titles whose real thread ids are represented in this log window.
 * Missing keys are not deletions; an explicit empty title resets the override. */
export function boundedThreadTitles(titles: unknown, log: GroupMessage[]): Record<string, string> {
  if (!titles || typeof titles !== 'object' || Array.isArray(titles)) {
    return {}
  }

  const source = titles as Record<string, unknown>
  const ids = new Set(log.map(entry => entry.thread || 'legacy'))

  return Object.fromEntries([...ids].flatMap(id => {
    const title = Object.prototype.hasOwnProperty.call(source, id) ? source[id] : undefined

    return typeof title === 'string' ? [[id, boundedThreadTitle(title)]] : []
  }))
}
