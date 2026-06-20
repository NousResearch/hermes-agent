/**
 * native-init.ts — native chrome setup (status bar + nav bar icon styling).
 *
 * We draw the app edge-to-edge and inset content with CSS safe-area padding
 * (see theme-fallback.css), so the system bars OVERLAY the WebView and their
 * BACKGROUNDS stay transparent — the themed app background shows through behind
 * the clock/battery and the Android nav buttons. To keep those icons readable on
 * every theme we only flip their CONTENT brightness (dark icons on light themes,
 * light icons on dark themes), tracking the app's `.dark` class live.
 */
import { Capacitor } from '@capacitor/core'
import { SafeArea, SystemBarsStyle } from '@capacitor-community/safe-area'
import { StatusBar } from '@capacitor/status-bar'

export async function initNativeChrome(): Promise<void> {
  if (!Capacitor.isNativePlatform()) return
  try {
    await StatusBar.setOverlaysWebView({ overlay: true })
    syncSystemBars()
    observeThemeChanges()
  } catch {
    /* native plugins unavailable — CSS insets still apply */
  }
}

let themeObserver: MutationObserver | undefined

/**
 * Match BOTH system bars' icon brightness to the active app theme: dark icons on
 * a light theme, light icons on a dark theme. SafeArea is the single styler for
 * both bars (omitting `type` targets status bar AND nav bar) — using the
 * @capacitor/status-bar plugin alongside it lets SafeArea override the status bar
 * back to its config value. Backgrounds stay transparent, so the themed app
 * background continues behind both bars.
 */
export function syncSystemBars(): void {
  if (!Capacitor.isNativePlatform()) return
  const isDark = document.documentElement.classList.contains('dark')
  // SafeArea inverts the naming: "Light" = DARK icons (for a light background).
  void SafeArea.setSystemBarsStyle({
    style: isDark ? SystemBarsStyle.Dark : SystemBarsStyle.Light,
  }).catch(() => {})
}

/** Watch the desktop ThemeProvider's `.dark` toggle on <html> and re-sync. */
function observeThemeChanges(): void {
  if (themeObserver) return
  themeObserver = new MutationObserver(() => syncSystemBars())
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })
}
