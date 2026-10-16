import type { GroupMessage } from './types'

export const GROUP_THREAD_TITLE_CHARS = 120

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

    return typeof title === 'string' ? [[id, title.trim().slice(0, GROUP_THREAD_TITLE_CHARS)]] : []
  }))
}
