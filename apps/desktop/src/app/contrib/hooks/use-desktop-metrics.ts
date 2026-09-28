/**
 * Wires Desktop love/hate telemetry (store/desktop-metrics.ts) to the app's
 * existing state: the focused gateway + its collection switch, routes, open
 * overlays, workspace mode, user interaction, long frames and the profile
 * count. Main window only; renders nothing.
 */

import { useEffect } from 'react'

import { contributedRoutes } from '@/app/routes'
import { $workspaceMode } from '@/components/pane-shell/workspace-scope'
import { $commandPaletteOpen } from '@/store/command-palette'
import {
  bindDesktopMetrics,
  cancelPendingBackendDrop,
  noteInteraction,
  persistDesktopMetricsNow,
  recordFeatureUse,
  recordFriction,
  routeArea,
  setDesktopBotCount,
  setDesktopMetricsGate,
  slowFrameBucket,
  tickDesktopMetrics,
  trackArea,
  trackFlow
} from '@/store/desktop-metrics'
import { $findInPage } from '@/store/find-in-page'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { SESSION_SEARCH_FOCUS_EVENT } from '@/store/layout'
import { $profiles, $profilesByConnection } from '@/store/profile'
import { $modelPickerOpen, $sessionPickerOpen } from '@/store/session'
import { $switcherOpen } from '@/store/session-switcher'
import {
  $sharedMetricsConsent,
  readSharedMetricsConsent,
  type SharedMetricsConsent,
  type SharedMetricsRequester
} from '@/store/shared-metrics'

import { observeOnboardingMetrics } from './desktop-onboarding-metrics'

const DAY_TICK_MS = 10 * 60_000
const INTERACTION_THROTTLE_MS = 1000

function gateFor(consent: SharedMetricsConsent | null): 'off' | 'on' | null {
  return consent ? (consent.enabled ? 'on' : 'off') : null
}

function botCount(): number {
  const byConnection = [...$profilesByConnection.get().values()].reduce((sum, list) => sum + list.length, 0)

  return byConnection || $profiles.get().length
}

/** Long frames (Long Animation Frames where available, else long tasks) while the window is visible. */
function observeSlowFrames(): () => void {
  if (typeof PerformanceObserver === 'undefined') {
    return () => undefined
  }

  const types = PerformanceObserver.supportedEntryTypes ?? []
  const type = types.includes('long-animation-frame') ? 'long-animation-frame' : types.includes('longtask') ? 'longtask' : null

  if (!type) {
    return () => undefined
  }

  const observer = new PerformanceObserver(list => {
    if (document.visibilityState !== 'visible') {
      return
    }

    for (const entry of list.getEntries()) {
      const bucket = slowFrameBucket(entry.duration)

      if (bucket) {
        recordFriction('slow_frame', bucket)
      }
    }
  })

  try {
    observer.observe({ buffered: false, type })
  } catch {
    return () => undefined
  }

  return () => observer.disconnect()
}

export function useDesktopMetrics({
  enabled,
  gatewayOpen,
  pathname,
  profile,
  requestGateway
}: {
  enabled: boolean
  gatewayOpen: boolean
  pathname: string
  profile: string
  requestGateway: SharedMetricsRequester
}): void {
  // The focused gateway and its collection switch (re-read on every attach and profile switch).
  useEffect(() => {
    if (!enabled || !gatewayOpen) {
      bindDesktopMetrics(null)

      return
    }

    let cancelled = false

    bindDesktopMetrics(requestGateway)
    void readSharedMetricsConsent(requestGateway).then(consent => {
      // An unreadable answer (older backend, flap) keeps the last known gate.
      if (!cancelled && consent) {
        setDesktopMetricsGate(gateFor(consent))
      }
    })

    return () => {
      cancelled = true
    }
  }, [enabled, gatewayOpen, profile, requestGateway])

  // A first-run answer or a Settings change lands in the consent atom.
  useEffect(() => {
    if (!enabled) {
      return
    }

    return $sharedMetricsConsent.listen(consent => {
      if (consent) {
        setDesktopMetricsGate(gateFor(consent))
      }
    })
  }, [enabled])

  useEffect(() => {
    const area = enabled ? routeArea(pathname, contributedRoutes().map(route => route.path)) : null

    if (area) {
      recordFeatureUse(area)
    }
  }, [enabled, pathname])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const overlay = (area: 'command_palette' | 'model_picker' | 'session_picker' | 'session_switcher') => (open: boolean) => {
      if (open) {
        recordFeatureUse(area)
      }

      trackFlow(area, open)
    }

    let findActive = $findInPage.get().active
    const onSessionSearch = () => recordFeatureUse('session_search')

    const onInteraction = (() => {
      let last = 0

      return () => {
        const now = Date.now()

        if (now - last >= INTERACTION_THROTTLE_MS) {
          last = now
          noteInteraction(now)
        }
      }
    })()

    const onVisibility = () => {
      if (document.visibilityState === 'hidden') {
        persistDesktopMetricsNow()
      } else {
        tickDesktopMetrics()
      }
    }

    const stops = [
      $commandPaletteOpen.listen(overlay('command_palette')),
      $modelPickerOpen.listen(overlay('model_picker')),
      $sessionPickerOpen.listen(overlay('session_picker')),
      $switcherOpen.listen(overlay('session_switcher')),
      $findInPage.listen(next => {
        if (next.active && !findActive) {
          recordFeatureUse('find_in_page')
        }

        findActive = next.active
      }),
      // Bot Mode is the `bots` workspace (components/pane-shell/workspace-scope.ts).
      $workspaceMode.listen(mode => trackArea('bot_mode', mode === 'bots')),
      $gatewaySwitching.listen(switching => switching && cancelPendingBackendDrop()),
      $profiles.listen(() => setDesktopBotCount(botCount())),
      $profilesByConnection.listen(() => setDesktopBotCount(botCount())),
      observeSlowFrames(),
      observeOnboardingMetrics()
    ]

    setDesktopBotCount(botCount())
    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, onSessionSearch)
    window.addEventListener('pointerdown', onInteraction, { capture: true, passive: true })
    window.addEventListener('keydown', onInteraction, { capture: true, passive: true })
    window.addEventListener('wheel', onInteraction, { capture: true, passive: true })
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('pagehide', persistDesktopMetricsNow)
    const tick = window.setInterval(() => tickDesktopMetrics(), DAY_TICK_MS)

    return () => {
      stops.forEach(stop => stop())
      window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, onSessionSearch)
      window.removeEventListener('pointerdown', onInteraction, { capture: true })
      window.removeEventListener('keydown', onInteraction, { capture: true })
      window.removeEventListener('wheel', onInteraction, { capture: true })
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('pagehide', persistDesktopMetricsNow)
      window.clearInterval(tick)
    }
  }, [enabled])
}
