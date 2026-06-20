/**
 * i18n-overrides — mobile-only wording tweaks.
 *
 * The mobile app bundles the desktop renderer and shares its i18n catalog. A few
 * strings describe mouse/keyboard gestures that don't exist on a phone, so we
 * override them here for touch. The I18nProvider reads `TRANSLATIONS[locale]` by
 * reference (no clone), so mutating the catalog before the app renders is enough.
 * This is imported only by the mobile entry — the desktop build is untouched.
 */
import { TRANSLATIONS } from '@/i18n'

// Pinning is shift-click on desktop; on a phone it's a long-press.
TRANSLATIONS.en.sidebar.shiftClickHint = 'Press and hold a chat to pin'

// Boot-failure screen: this client only ever talks to a REMOTE gateway — there's
// no local install/gateway to repair or fall back to (those buttons are hidden in
// theme-fallback.css). Reword so the copy doesn't reference a local backend.
TRANSLATIONS.en.boot.failure.title = "Can't reach the gateway"
TRANSLATIONS.en.boot.failure.description =
  "Couldn't connect to your gateway. Check that it's running and reachable, then retry. Nothing here is deleted."
TRANSLATIONS.en.boot.failure.repairHint =
  'Make sure your gateway is running and reachable, then try again.'
TRANSLATIONS.en.boot.failure.remoteSignInHint = 'Opens the gateway login window to reconnect.'

// Boot-failure TOASTS (rendered via translateNow, which reads the catalog live):
// drop the "Desktop" wording — this is a phone app talking to a remote gateway.
TRANSLATIONS.en.boot.errors.desktopBootFailed = "Couldn't reach the gateway"
TRANSLATIONS.en.boot.desktopBootFailedWithMessage = (message: string) =>
  `Couldn't reach the gateway: ${message}`
