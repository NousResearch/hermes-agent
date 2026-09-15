type ZoomState = { level: number; percent: number }

/** App scale is independent of browser zoom and monitor pixel density. */
export function createBrowserZoom(basePath: string): NonNullable<Window['hermesDesktop']['zoom']> {
  const storageKey = `hermes:browser:ui-scale:${basePath || '/'}`
  const listeners = new Set<(state: ZoomState) => void>()
  let percent = 100

  const normalize = (value: number) =>
    Number.isFinite(value) && value > 0 ? Math.round(Math.min(200, Math.max(50, value))) : 100

  const state = (): ZoomState => ({ level: Math.log(percent / 100) / Math.log(1.2), percent })

  const apply = (value: number) => {
    percent = normalize(value)
    document.documentElement.style.zoom = String(percent / 100)
  }

  try {
    apply(Number(localStorage.getItem(storageKey)) || 100)
  } catch {
    // Private browsing can deny storage; the current tab can still scale.
    apply(100)
  }

  const setPercent = (value: number) => {
    apply(value)

    try {
      localStorage.setItem(storageKey, String(percent))
    } catch {
      // Keep the applied value even when persistence is unavailable.
    }

    listeners.forEach(callback => callback(state()))
    // Layout consumers such as terminal fit need to remeasure at the new scale.
    window.dispatchEvent(new Event('resize'))
  }

  return {
    get: async () => state(),
    factor: () => percent / 100,
    onChanged: callback => {
      listeners.add(callback)

      return () => {
        listeners.delete(callback)
      }
    },
    setPercent
  }
}
