import { useEffect, useState } from 'react'

import { GatewaySettings } from '@/app/settings/gateway-settings'

export function RemoteGatewaySetupOverlay() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const desktop = window.hermesDesktop

    if (!desktop?.getDesktopCapabilities || !desktop.getConnectionConfig) {
      return
    }

    let cancelled = false

    void Promise.all([desktop.getDesktopCapabilities(), desktop.getConnectionConfig()])
      .then(([capabilities, config]) => {
        if (!cancelled) {
          setVisible(capabilities.remoteOnly && config.mode === 'local')
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [])

  if (!visible) {
    return null
  }

  return (
    <div className="fixed inset-0 z-[1500] flex items-center justify-center bg-(--ui-chat-surface-background) p-4">
      <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) pt-6 shadow-nous">
        <GatewaySettings onConnected={() => setVisible(false)} remoteOnly />
      </div>
    </div>
  )
}
