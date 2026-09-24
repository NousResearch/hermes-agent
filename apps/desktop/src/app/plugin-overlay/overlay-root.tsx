import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { useGatewayBoot } from '@/app/gateway/hooks/use-gateway-boot'
import { loadOverlayPluginById } from '@/contrib/runtime-loader'
import { $gatewayState } from '@/store/session'
import { ThemeProvider } from '@/themes/context'
import { useStore } from '@nanostores/react'

import { PluginOverlayApp } from './plugin-overlay-app'

/**
 * Boot the plugin-overlay window (`?win=plugoverlay&plugin=<id>`).
 *
 * A full-app renderer like the HUD — it boots its OWN gateway so the hosted
 * plugin's `host.request` / `ctx.rest` keep working with the main window
 * minimized — but with a bespoke surface: no app shell, no pane tree, just
 * the one `area: 'pluginOverlay'` contribution of the hosted plugin.
 *
 * Boot sequence (the design intent): whoami → loadOverlayPluginById →
 * registry read → render. The gateway boot reuses the shared
 * `useGatewayBoot` hook (the HUD's pattern) rather than a hand-rolled
 * connection, so reconnect/backoff/profile adoption behave identically.
 * Only the app-level consumers are stubbed: the overlay has no session
 * list, no config surface, no server-request handlers to answer.
 *
 * The index.html boot script paints an OPAQUE themed background to avoid a
 * flash in normal windows; the overlay must be see-through, so we force
 * every host layer transparent with a late, high-specificity style tag (the
 * pet overlay pattern).
 */
export function mountPluginOverlay(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="plugin-overlay">
        <ThemeProvider>
          <PluginOverlayBoot />
        </ThemeProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}

/**
 * Boots the gateway, asks main which plugin this window hosts and in which
 * posture (mascot sprite vs interactive card), loads that plugin, then hands
 * the contribution to PluginOverlayApp. Main owns posture flips: it pushes
 * them over `onMode` and the boot re-reads the current one via `whoami`.
 */
function PluginOverlayBoot() {
  const [pluginId, setPluginId] = useState<null | string>(null)
  const [mode, setMode] = useState<null | 'mascot' | 'card'>(null)
  const [failed, setFailed] = useState<null | string>(null)
  const loadedRef = useRef(false)

  useGatewayBoot({
    beforeConnectionSwitch: () => {},
    handleGatewayEvent: () => {},
    // No server→client requests to answer in the overlay — every channel
    // answers -32601 (the registry's fail path).
    handleServerRequest: () => false,
    onConnectionReady: () => {},
    onGatewayReady: () => {},
    // The overlay hosts no session/config surface; these keep the boot
    // contract but do not pull app data.
    refreshHermesConfig: async () => {},
    refreshSessions: async () => {}
  })

  const gatewayState = useStore($gatewayState)

  // Posture flips (mascot click → card, card ✕ → mascot) arrive from main
  // once the overlay is live.
  useEffect(() => {
    const bridge = window.hermesDesktop?.pluginOverlay

    return bridge?.onMode(setMode)
  }, [])

  useEffect(() => {
    if (loadedRef.current || failed || gatewayState !== 'open') {
      return
    }

    loadedRef.current = true
    let cancelled = false

    void (async () => {
      try {
        const bridge = window.hermesDesktop?.pluginOverlay

        if (!bridge) {
          if (!cancelled) setFailed('The plugin-overlay bridge is unavailable.')

          return
        }

        const who = await bridge.whoami()

        if (cancelled) return

        if (!who.pluginId) {
          setFailed('This overlay window has no hosted plugin.')

          return
        }

        if (!(await loadOverlayPluginById(who.pluginId))) {
          if (!cancelled) setFailed(`Plugin "${who.pluginId}" could not be loaded.`)

          return
        }

        if (!cancelled) {
          setPluginId(who.pluginId)
          setMode(who.mode === 'mascot' ? 'mascot' : 'card')
        }
      } catch (err) {
        if (!cancelled) {
          setFailed(`Overlay boot failed: ${err instanceof Error ? err.message : String(err)}`)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [failed, gatewayState])

  if (failed) {
    return (
      <div
        style={{
          alignItems: 'center',
          color: 'var(--ui-text-secondary)',
          display: 'flex',
          height: '100vh',
          justifyContent: 'center',
          padding: 16,
          textAlign: 'center'
        }}
      >
        {failed}
      </div>
    )
  }

  return <PluginOverlayApp pluginId={pluginId} mode={mode} ready={gatewayState === 'open'} />
}
