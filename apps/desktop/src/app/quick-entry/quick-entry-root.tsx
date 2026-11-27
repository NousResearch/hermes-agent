import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { I18nProvider, isSupportedLocaleValue, normalizeLocale, useI18n } from '@/i18n'
import { ThemeProvider } from '@/themes/context'

import { QuickEntryApp } from './quick-entry-app'

/**
 * The card has no gateway, so no `display.language` read: it speaks the
 * language the primary window relays with every state push. With no config
 * client, `setLocale` only switches the card; nothing is saved.
 */
function QuickEntryLocaleSync() {
  const { locale, setLocale } = useI18n()

  useEffect(
    () =>
      window.hermesDesktop?.quickEntry?.onState(payload => {
        const next = payload?.locale

        if (isSupportedLocaleValue(next) && normalizeLocale(next) !== locale) {
          void setLocale(normalizeLocale(next))
        }
      }),
    [locale, setLocale]
  )

  return null
}

/**
 * Boot the Quick Entry window. Loaded by the same bundle as the main app but via
 * `?win=quick`, so it shares CSS/theme tokens while mounting a minimal capture
 * surface (no app shell, no gateway, no router).
 *
 * The index.html boot script paints an OPAQUE themed background to avoid a flash
 * in normal windows; this window is a floating card on a transparent backdrop,
 * so force the host layers see-through (same trick as the pet overlay).
 */
export function mountQuickEntry(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="quick-entry">
        <I18nProvider configClient={null}>
          <QuickEntryLocaleSync />
          <ThemeProvider>
            <QuickEntryApp />
          </ThemeProvider>
        </I18nProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}
