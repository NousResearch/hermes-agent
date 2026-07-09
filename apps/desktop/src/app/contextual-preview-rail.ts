import type { AppView } from './routes'

export interface ContextualPreviewRailState {
  currentView: AppView | string
  hasPreviewTarget: boolean
  routedSessionId: null | string | undefined
  selectedStoredSessionId: null | string | undefined
}

export function contextualPreviewSessionId({
  routedSessionId,
  selectedStoredSessionId
}: Pick<ContextualPreviewRailState, 'routedSessionId' | 'selectedStoredSessionId'>): string {
  return selectedStoredSessionId?.trim() || routedSessionId?.trim() || ''
}

export function shouldShowContextualPreviewRail(state: ContextualPreviewRailState): boolean {
  return state.currentView === 'chat' && state.hasPreviewTarget && Boolean(contextualPreviewSessionId(state))
}
