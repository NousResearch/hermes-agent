export type BusyInputMode = 'interrupt' | 'queue' | 'steer'
export type BusyComposerAction = 'redirect' | 'queue' | 'steer' | 'stop'

export const normalizeBusyInputMode = (value: unknown): BusyInputMode =>
  value === 'queue' || value === 'steer' ? value : 'interrupt'

/**
 * Maps the persisted cross-surface busy-input policy to Desktop's visible
 * composer action. `interrupt` is the historical config spelling; capable
 * desktop sessions implement it as an active-turn redirect.
 */
export function resolveBusyComposerAction({
  busy,
  canRedirect,
  canSteer,
  compacting,
  hasPayload,
  mode
}: {
  busy: boolean
  canRedirect: boolean
  canSteer: boolean
  compacting: boolean
  hasPayload: boolean
  mode: BusyInputMode
}): BusyComposerAction {
  if (!busy) {
    return 'stop'
  }

  if (mode === 'queue') {
    return hasPayload ? 'queue' : 'stop'
  }

  if (!compacting && mode === 'interrupt' && canRedirect) {
    return 'redirect'
  }

  if (!compacting && mode === 'steer' && canSteer) {
    return 'steer'
  }

  return hasPayload ? 'queue' : 'stop'
}
