import { lazy, Suspense, useEffect, useState } from 'react'

import { Loader } from '@/components/ui/loader'
import type { DesktopCapabilities } from '@/global'
import { useI18n } from '@/i18n'

const GatewaySettings = lazy(() =>
  import('@/app/settings/gateway-settings').then(module => ({ default: module.GatewaySettings }))
)

/**
 * The remote-only setup gate is controlled by main-process state, not by the
 * persisted mode alone. A malformed URL/token still has a remote mode in the
 * file, but startup cannot resolve it; `remoteSetupRequired` is the durable
 * signal that keeps this recovery route visible.
 */
export function shouldShowRemoteGatewaySetup(capabilities: DesktopCapabilities | null | undefined): boolean {
  return capabilities?.remoteOnly === true && capabilities.remoteSetupRequired === true
}

export function remoteSetupErrorText(
  capabilities: DesktopCapabilities | null | undefined,
  label: string
): null | string {
  if (!shouldShowRemoteGatewaySetup(capabilities) || !capabilities?.remoteSetupError) {
    return null
  }

  return `${label}: ${capabilities.remoteSetupError}`
}

export function RemoteGatewaySetupOverlay() {
  const { t } = useI18n()
  const copy = t.settings.gateway
  const [capabilities, setCapabilities] = useState<DesktopCapabilities | null>(null)

  useEffect(() => {
    const desktop = window.hermesDesktop

    if (!desktop?.getDesktopCapabilities) {
      return
    }

    let cancelled = false

    const onCapabilities = desktop.onDesktopCapabilities?.(next => {
      if (!cancelled) {
        setCapabilities(next)
      }
    })

    void desktop
      .getDesktopCapabilities()
      .then(nextCapabilities => {
        if (cancelled) {
          return
        }

        setCapabilities(nextCapabilities)
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
      onCapabilities?.()
    }
  }, [])

  if (!shouldShowRemoteGatewaySetup(capabilities)) {
    return null
  }

  const setupErrorText = remoteSetupErrorText(capabilities, copy.testFailed)

  return (
    <div
      className="fixed inset-0 z-(--z-setup) flex items-center justify-center bg-(--ui-chat-surface-background) p-4"
      data-glass-opaque=""
    >
      <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) pt-6 shadow-nous">
        <div className="px-6 pb-2">
          <h1 className="text-lg font-semibold">{copy.title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{copy.remoteDesc}</p>
          {setupErrorText ? (
            <p className="mt-2 text-sm text-destructive" role="alert">
              {setupErrorText}
            </p>
          ) : null}
        </div>
        <Suspense fallback={<Loader className="mx-auto my-16 size-6 text-(--ui-text-tertiary)" />}>
          <GatewaySettings embedded onConnected={() => setCapabilities(null)} remoteOnly />
        </Suspense>
      </div>
    </div>
  )
}
