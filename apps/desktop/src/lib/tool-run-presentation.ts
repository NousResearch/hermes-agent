import type { Contribution } from '@/contrib/types'

/** Optional presentation-only preference, consumed by native ordinary tool runs. */
export const TOOL_RUN_PRESENTATION_API_VERSION = 1
export const TOOL_RUN_PRESENTATION_AREA = 'chat.toolRun.presentation'

export type ToolRunPresentationMode = 'grouped' | 'individual'

export interface ToolRunPresentationContribution {
  readonly apiVersion: typeof TOOL_RUN_PRESENTATION_API_VERSION
  readonly mode: ToolRunPresentationMode
}

/** Lowest finite order (default 0), then ordinal namespaced id, wins.
 * Invalid payloads, including throwing getters, are ignored. No valid preference
 * means native grouped presentation. Read only on registry snapshot changes;
 * replace the contribution to change mode, and dispose it on disable/unload.
 */
export function resolveToolRunPresentation(entries: readonly Contribution[]): ToolRunPresentationMode {
  const candidates: { id: string; mode: ToolRunPresentationMode; order: number }[] = []

  for (const entry of entries) {
    try {
      const data = entry.data as Partial<ToolRunPresentationContribution> | null | undefined
      const mode = data?.mode
      const order = entry.order ?? 0

      if (
        data?.apiVersion === TOOL_RUN_PRESENTATION_API_VERSION &&
        (mode === 'grouped' || mode === 'individual') &&
        Number.isFinite(order)
      ) {
        candidates.push({ id: entry.id, mode, order })
      }
    } catch {
      // Runtime contributions are not necessarily well-typed. Stay native.
    }
  }

  candidates.sort((a, b) => a.order - b.order || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))

  return candidates[0]?.mode ?? 'grouped'
}
