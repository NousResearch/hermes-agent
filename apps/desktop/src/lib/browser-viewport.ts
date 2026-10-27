/** Fit the browser app to the visible area when a keyboard leaves layout height
 * unchanged. `interactive-widget=resizes-content` handles supporting browsers;
 * VisualViewport covers the others. Pan/zoom stays with the browser: a pinch
 * must not turn its smaller visible area into a smaller application layout.
 */
export function installBrowserViewport(): () => void {
  const viewport = window.visualViewport
  const root = document.getElementById('root')

  if (!viewport || !root || document.documentElement.dataset.hermesDesktopHost !== 'browser') {
    return () => {}
  }

  let frame = 0
  let unzoomedTop = 0

  const reset = () => {
    root.removeAttribute('data-browser-viewport')
    root.style.removeProperty('--browser-viewport-height')
    root.style.removeProperty('--browser-viewport-top')
  }

  const update = () => {
    frame = 0

    // Remove pinch magnification from the height, so opening/closing the
    // keyboard still updates available space while zoomed. No focus(),
    // scrollIntoView(), or browser zoom changes here.
    const height = viewport.height * viewport.scale

    if (!Number.isFinite(height) || height <= 0) {
      return
    }

    // iOS Safari reports the visible area as innerHeight, zoomed or not, and
    // scrolls the focused field into view itself. Fitting the shell under that
    // scroll would shrink the app out of the visible area.
    if (Math.abs(viewport.height - window.innerHeight) < 1) {
      unzoomedTop = 0
      reset()

      return
    }

    // Offset during a pinch belongs to the user's pan, not the app shell.
    if (Math.abs(viewport.scale - 1) < 0.01) {
      unzoomedTop = Math.max(0, viewport.offsetTop)
    }

    const top = Math.min(unzoomedTop, Math.max(0, window.innerHeight - height))

    if (Math.abs(height - window.innerHeight) < 1 && top < 1) {
      unzoomedTop = 0
      reset()

      return
    }

    // Hermes' UI scale is CSS zoom on <html>, separate from pinch zoom.
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1
    root.style.setProperty('--browser-viewport-height', `${height / zoom}px`)
    root.style.setProperty('--browser-viewport-top', `${top / zoom}px`)
    root.setAttribute('data-browser-viewport', '')
  }

  const schedule = () => {
    if (!frame) {
      frame = requestAnimationFrame(update)
    }
  }

  viewport.addEventListener('resize', schedule)
  viewport.addEventListener('scroll', schedule)
  window.addEventListener('resize', schedule)
  update()

  return () => {
    cancelAnimationFrame(frame)
    viewport.removeEventListener('resize', schedule)
    viewport.removeEventListener('scroll', schedule)
    window.removeEventListener('resize', schedule)
    reset()
  }
}
