import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { getSharedMetricsConsent, setSharedMetricsConsent } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingGate, guidedOnboardingActive } from '@/store/onboarding-gate'

import { invalidateSharedMetricsConsent } from '../app/settings/telemetry-settings'

// Profiles already asked this app run. A failed read or a dismissed dialog must not
// re-ask on the next re-render / reconnect — once per launch, then the backend's global
// `decided` answer takes over for every profile.
const askedThisRun = new Set<string>()

/**
 * One-time "share anonymous usage metrics?" question, asked when the active profile
 * has telemetry off AND nobody has ever answered (the backend's global `decided` is
 * false). The answer is recorded once at the Hermes root and inherited by every
 * profile, so this shows at most once per user. Onboarding records an answer on
 * every route out of the picker, so a fresh install never sees this — it exists for
 * upgraded installs and setups done outside the desktop. Either button, Esc or a
 * backdrop click records an answer.
 */
export function TelemetryConsentPrompt({ enabled, profile }: { enabled: boolean; profile: string }) {
  const { t } = useI18n()
  const copy = t.settings.telemetry
  const onboarding = useStore($desktopOnboarding)
  // Subscribed only so a phase change re-evaluates `settled`.
  useStore($onboardingGate)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)

  // Wait for the backend to be reachable and every onboarding surface to be out of the
  // way: the picker (which answers this itself), the free-tier ready screen, and the
  // guided first launch. Never stack this dialog on top of any of them.
  const settled =
    enabled &&
    onboarding.configured === true &&
    !onboarding.requested &&
    !onboarding.manual &&
    !onboarding.freeTierReady &&
    !guidedOnboardingActive()

  useEffect(() => {
    if (!settled || askedThisRun.has(profile)) {
      return
    }

    askedThisRun.add(profile)
    let cancelled = false
    void getSharedMetricsConsent(profile)
      .then(state => {
        if (!cancelled && !state.decided && !state.enabled) {
          setOpen(true)
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [profile, settled])

  const answer = async (share: boolean) => {
    if (busy) {
      return
    }

    setBusy(true)

    try {
      await setSharedMetricsConsent({ enabled: share, send: share }, profile)
      void invalidateSharedMetricsConsent(profile)
    } catch (error) {
      notifyError(error, copy.failedSave)
    } finally {
      setBusy(false)
      setOpen(false)
    }
  }

  if (!open) {
    return null
  }

  return (
    <ConfirmDialog
      cancelLabel={copy.promptDecline}
      confirmLabel={copy.promptAccept}
      description={copy.promptDescription}
      dismissOnConfirm
      onClose={() => void answer(false)}
      onConfirm={() => answer(true)}
      open={open}
      title={copy.promptTitle}
    />
  )
}
