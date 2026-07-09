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

export interface ContextualPreviewPaneState extends ContextualPreviewRailState {
  paneOpen: boolean
}

export function nextContextualPreviewPaneOpen(state: ContextualPreviewPaneState): boolean {
  return shouldShowContextualPreviewRail(state)
}

export function shouldShowContextualPreviewRail(state: ContextualPreviewRailState): boolean {
  return state.currentView === 'chat' && state.hasPreviewTarget && Boolean(contextualPreviewSessionId(state))
}
