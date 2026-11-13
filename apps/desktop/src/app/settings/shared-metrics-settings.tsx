import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { DocsLink } from '@/components/onboarding/flow'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { captureDesktopMetricsGate } from '@/store/desktop-metrics'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $settingsScopeProfile } from '@/store/settings-scope'
import {
  readSharedMetricsConsent,
  saveSharedMetricsConsent,
  SHARED_METRICS_DOCS_URL,
  type SharedMetricsConsent,
  sharedMetricsProfileRequester
} from '@/store/shared-metrics'

import { ToggleRow } from './primitives'

/**
 * Settings › Safety › Privacy: one switch for the answer the first-run offer and
 * `hermes setup` write, for the profile this page applies to. On sets both
 * opt-ins (collect and send), off clears both. A collect-only answer from
 * before the offer dropped it reads off here: nothing is shared. Optimistic,
 * rolled back on failure.
 */
export function SharedMetricsSettings() {
  const profile = useStore($settingsScopeProfile)
  const connectionId = useStore($activeConnectionId)

  return (
    <SharedMetricsSettingsForOwner
      connectionId={connectionId}
      key={JSON.stringify([connectionId, profile])}
      profile={profile}
    />
  )
}

interface SharedMetricsOwnerProps {
  connectionId: string | null
  profile: string
}

function SharedMetricsSettingsForOwner({ connectionId, profile }: SharedMetricsOwnerProps) {
  const { t } = useI18n()
  const copy = t.sharedMetrics
  const [consent, setConsent] = useState<SharedMetricsConsent | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [busy, setBusy] = useState(false)

  const request = useMemo(
    () =>
      sharedMetricsProfileRequester(
        <T,>(method: string, params: Record<string, unknown> = {}) =>
          requestGatewayForAgent<T>(connectionId, profile, method, params, undefined, undefined, {
            spawnPriority: 'foreground'
          }),
        profile
      ),
    [connectionId, profile]
  )

  useEffect(() => {
    let cancelled = false

    setLoaded(false)
    void readSharedMetricsConsent(request).then(next => {
      if (!cancelled) {
        setConsent(next)
        setLoaded(true)
      }
    })

    return () => void (cancelled = true)
  }, [request])

  const save = async (flags: { enabled: boolean; send: boolean }) => {
    const previous = consent

    setBusy(true)
    setConsent({ ...flags, send: flags.enabled && flags.send, decided: true })

    try {
      const applyGate = await captureDesktopMetricsGate(`${connectionId ?? ''}|${profile}`)
      const saved = await saveSharedMetricsConsent(request, flags)

      setConsent(saved)

      await applyGate(saved.enabled ? 'on' : 'off')
    } catch (err) {
      setConsent(previous)
      notifyError(err, copy.saveFailed)
    } finally {
      setBusy(false)
    }
  }

  // An older backend has no shared_metrics.* methods: keep the control, disabled and explained.
  const unavailable = loaded && consent === null
  const sharing = consent?.send ?? false

  return (
    <div className="grid gap-1" id="setting-shared-metrics">
      <ToggleRow
        below={<DocsLink href={SHARED_METRICS_DOCS_URL}>{copy.whatIsCollected}</DocsLink>}
        checked={sharing}
        description={copy.sendDesc}
        disabled={!consent || busy}
        hint={unavailable ? copy.unavailable : undefined}
        label={copy.sendLabel}
        onChange={on => void save({ enabled: on, send: on })}
      />
    </div>
  )
}
