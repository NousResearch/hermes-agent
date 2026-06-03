import { atom } from 'nanostores'

export interface DesktopZoomState {
  factor: number
  percent: number
}

const DEFAULT_ZOOM_STATE: DesktopZoomState = { factor: 1, percent: 100 }

export const $zoomState = atom<DesktopZoomState>(DEFAULT_ZOOM_STATE)

const applyZoomState = (state: DesktopZoomState | null | undefined) => {
  if (!state || typeof state.factor !== 'number' || typeof state.percent !== 'number') {
    return DEFAULT_ZOOM_STATE
  }

  const next = { factor: state.factor, percent: state.percent }
  $zoomState.set(next)
  return next
}

export async function initializeZoom() {
  const api = window.hermesDesktop?.zoom
  if (!api) return DEFAULT_ZOOM_STATE

  const unsubscribe = api.onChanged?.(applyZoomState)
  const current = await api.get().catch(() => DEFAULT_ZOOM_STATE)
  applyZoomState(current)

  return { ...$zoomState.get(), unsubscribe }
}

export async function zoomIn() {
  const next = await window.hermesDesktop?.zoom?.adjust(1)
  return applyZoomState(next)
}

export async function zoomOut() {
  const next = await window.hermesDesktop?.zoom?.adjust(-1)
  return applyZoomState(next)
}

export async function resetZoom() {
  const next = await window.hermesDesktop?.zoom?.reset()
  return applyZoomState(next)
}
