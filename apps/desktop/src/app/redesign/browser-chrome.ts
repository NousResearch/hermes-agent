import { isBrowserHostedDesktop } from '@/lib/platform'
import { $portalShell } from '@/store/shell-variant'

const SAFE_AREA = 'viewport-fit=cover'

/** The redesign's share of the mobile browser's own chrome, on the browser host only (native windows
 *  have none):
 *  - `<meta name="theme-color">` follows the redesign's canvas in the current theme, so the browser's
 *    toolbar and status bar continue the band instead of the boot colour;
 *  - the viewport extends under the notch and home indicator (`viewport-fit=cover`); on a device
 *    that reports a safe area, `<html data-portal-safe-area>` insets the app root by it
 *    (portal-shell.css) and body-portalled surfaces read `--portal-safe-*` (tokens/space.css).
 *  Classic keeps index.html's values: both are restored when the shell flips back. */
export function installPortalBrowserChrome(root: HTMLElement = document.documentElement): () => void {
  const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
  const viewportMeta = document.querySelector<HTMLMetaElement>('meta[name="viewport"]')

  if (!isBrowserHostedDesktop() || !themeMeta || !viewportMeta) {
    return () => {}
  }

  const classicTheme = themeMeta.content
  // Measures env(safe-area-inset-*): the root is inset only on a device that reports one.
  const safeArea = document.createElement('span')
  safeArea.hidden = true
  safeArea.style.padding =
    'env(safe-area-inset-top, 0px) env(safe-area-inset-right, 0px) env(safe-area-inset-bottom, 0px) env(safe-area-inset-left, 0px)'
  const classicViewport = viewportMeta.content
  // Resolves the canvas token (a color-mix of skin seeds) to the plain colour a browser accepts.
  const probe = document.createElement('span')
  probe.hidden = true
  probe.style.backgroundColor = 'var(--portal-surface-canvas)'
  let frame = 0

  const sync = () => {
    frame = 0

    if (!$portalShell.get()) {
      restore()

      return
    }

    if (!probe.isConnected) {
      document.body.append(probe, safeArea)
    }

    themeMeta.content = getComputedStyle(probe).backgroundColor
    viewportMeta.content = classicViewport.includes(SAFE_AREA) ? classicViewport : `${classicViewport}, ${SAFE_AREA}`
    const inset = getComputedStyle(safeArea)

    const any = [inset.paddingTop, inset.paddingRight, inset.paddingBottom, inset.paddingLeft].some(
      v => parseFloat(v) > 0
    )

    root.toggleAttribute('data-portal-safe-area', any)
  }

  const restore = () => {
    themeMeta.content = classicTheme
    viewportMeta.content = classicViewport
    probe.remove()
    safeArea.remove()
    root.removeAttribute('data-portal-safe-area')
  }

  const schedule = () => {
    frame ||= requestAnimationFrame(sync)
  }

  // Theme, mode and skin all land on <html> (class, data-theme, inline skin seeds).
  const observer = new MutationObserver(schedule)
  observer.observe(root, { attributeFilter: ['class', 'style', 'data-theme', 'data-shell'], attributes: true })
  const stopShell = $portalShell.listen(schedule)
  // Rotation moves the notch from the top to a side.
  window.addEventListener('resize', schedule)
  sync()

  return () => {
    cancelAnimationFrame(frame)
    observer.disconnect()
    stopShell()
    window.removeEventListener('resize', schedule)
    restore()
  }
}
