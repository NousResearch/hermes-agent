import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { applyAppearance, THEME_MODES } from '@/store/appearance-apply'
import { freeTierSetupFailure } from '@/store/free-tier'
import {
  $desktopOnboarding,
  completeDesktopOnboarding,
  type OnboardingContext,
  startManualOnboarding
} from '@/store/onboarding'
import { captureOnboardingScope, requestOnboardingGateway } from '@/store/onboarding-scope'
import {
  $guideRecovery,
  $guideScope,
  $guideStep,
  finishOnboarding,
  reconcileOnboarding,
  resumeOnboarding,
  saveGuideStep
} from '@/store/setup-guide'
import { answerSharedMetricsOffer, type SharedMetricsChoice, type SharedMetricsConsent } from '@/store/shared-metrics'
import { $setupGuideOnboarding } from '@/store/shell-policy'
import { useTheme } from '@/themes/context'
import type { FreeTierStatus } from '@/types/hermes'

import { setupFailureCopy } from './free-tier-setup-notice'

/** A standing offer, not a modal: settings must stay usable during recovery or handoff. */
export function SetupGuideOffer() {
  const { t } = useI18n()
  const recovery = useStore($guideRecovery)
  const guideScope = useStore($guideScope)
  const scope = recovery?.scope ?? guideScope
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const retry = async () => {
    setBusy(true)
    setError(null)
    try {
      await (recovery ? recovery.retry() : resumeOnboarding())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section
      aria-label={t.firstRun.guide.title}
      className="fixed bottom-4 end-4 z-(--z-onboarding) grid max-w-sm gap-3 rounded-xl bg-popover p-4 text-popover-foreground shadow-nous"
    >
      <h2 className="font-medium">{t.firstRun.guide.title}</h2>
      {scope && (
        <p className="text-xs text-muted-foreground">
          {[scope.connectionId, scope.profile].filter(Boolean).join(' / ')}
        </p>
      )}
      {recovery && (
        <p className="text-sm" role="alert">
          {recovery.upgradeRequired ? t.firstRun.guide.upgradeRequired : recovery.message}
        </p>
      )}
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      <Button disabled={busy} onClick={() => void retry()}>
        {recovery ? t.common.retry : t.firstRun.guide.resume}
      </Button>
    </section>
  )
}

export function SetupGuide({ ctx }: { ctx: OnboardingContext }) {
  const { t } = useI18n()
  const { mode: selectedMode } = useTheme()
  const copy = t.firstRun.guide
  const step = useStore($guideStep)
  const manual = useStore($desktopOnboarding).manual
  const [scope] = useState(() => $guideScope.get() ?? captureOnboardingScope(ctx.scope ?? ctx.profile))

  const request = useMemo(
    () =>
      <T,>(method: string, params?: Record<string, unknown>) =>
        requestOnboardingGateway<T>(scope, method, params),
    [scope]
  )

  const [runtime, setRuntime] = useState<RuntimeReadinessResult | null>(null)
  const [free, setFree] = useState<FreeTierStatus | null>(null)
  const [consent, setConsent] = useState<SharedMetricsConsent | null>(null)
  const [busy, setBusy] = useState(false)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const check = useCallback(async () => {
    setChecking(true)

    try {
      const [readiness, status] = await Promise.allSettled([
        evaluateRuntimeReadiness(request, { defaultReason: copy.providerNeeded }),
        request<FreeTierStatus>('free_tier.status')
      ])

      setRuntime(readiness.status === 'fulfilled' ? readiness.value : null)
      setFree(status.status === 'fulfilled' ? status.value : null)

      const failures = [readiness, status].flatMap(result =>
        result.status === 'rejected'
          ? [result.reason instanceof Error ? result.reason.message : String(result.reason)]
          : []
      )

      setError(failures.join('\n') || null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setChecking(false)
    }
  }, [copy.providerNeeded, request])

  const act = async (action: () => Promise<unknown>) => {
    setBusy(true)
    setError(null)

    try {
      await action()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void request<SharedMetricsConsent>('shared_metrics.status')
      .then(setConsent)
      .catch(err => setError(err instanceof Error ? err.message : String(err)))
  }, [request])
  useEffect(() => {
    if (step === 'model' && !manual) {
      void check()
    }
  }, [step, manual, check])
  useEffect(() => {
    const reconcile = () => {
      void reconcileOnboarding().catch(err => setError(String(err)))
    }

    window.addEventListener('focus', reconcile)
    const timer = window.setInterval(reconcile, 5000)

    return () => {
      window.removeEventListener('focus', reconcile)
      window.clearInterval(timer)
    }
  }, [])

  const chooseConsent = (choice: SharedMetricsChoice) =>
    act(async () => {
      await answerSharedMetricsOffer(request, choice, () => {
        const active = captureOnboardingScope()

        return active.connectionId === scope.connectionId && active.profile === scope.profile
      })
      setConsent(await request<SharedMetricsConsent>('shared_metrics.status'))
    })

  const ready = runtime?.ready === true && runtime.source === 'runtime_check'
  const failure = freeTierSetupFailure(free)

  const provision = () =>
    act(async () => {
      // Never provision over an already working provider, including a rerun.
      const current = await evaluateRuntimeReadiness(request)

      if (!$setupGuideOnboarding.get()) {
        return
      }

      if (!current.ready) {
        await request('free_tier.provision')
      }
      await check()
    })

  const finish = () =>
    act(async () => {
      const current = await evaluateRuntimeReadiness(request, { defaultReason: copy.providerNeeded })

      if (!$setupGuideOnboarding.get()) {
        return
      }
      setRuntime(current)

      if (!current.ready || current.source !== 'runtime_check') {
        return
      }

      if (free?.notice_pending) {
        const ack = await request<{ acked: boolean }>('free_tier.ack_notice')

        if (!ack.acked) {
          throw new Error(t.freeTier.setupFailed.generic)
        }
      }

      const active = captureOnboardingScope()

      if (active.connectionId === scope.connectionId && active.profile === scope.profile) {
        completeDesktopOnboarding()
      }

      // Setup configures its captured owner; it never replaces the foreground
      // draft or navigation, including after a settings handoff or an agent swap.
      await finishOnboarding('done')
    })

  return (
    <Dialog open={!manual}>
      <DialogContent
        aria-describedby="setup-guide-description"
        bodyClassName="gap-6 p-6"
        className="z-(--z-onboarding) w-[calc(100vw_-_2rem)] max-w-[40rem]"
        onEscapeKeyDown={event => event.preventDefault()}
        onPointerDownOutside={event => event.preventDefault()}
        overlayClassName="z-[calc(var(--z-onboarding)-1)]"
        showCloseButton={false}
      >
        <div className="grid gap-2">
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription id="setup-guide-description">{copy.intro}</DialogDescription>
        </div>
        {step === 'preferences' ? (
          <div className="grid gap-5">
            <div className="grid gap-2">
              <h2 className="font-medium">{copy.appearance}</h2>
              <div className="flex flex-wrap gap-2">
                {THEME_MODES.map(mode => (
                  <Button
                    aria-pressed={mode === selectedMode}
                    key={mode}
                    onClick={() => applyAppearance({ mode })}
                    variant={mode === selectedMode ? 'default' : 'outline'}
                  >
                    {t.settings.modeOptions[mode].label}
                  </Button>
                ))}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">{copy.privacy}</p>
            {consent?.decided === false && (
              <div className="grid gap-3">
                <h2 className="font-medium">{t.sharedMetrics.consentTitle}</h2>
                <p className="text-sm text-muted-foreground">{t.sharedMetrics.stripBody}</p>
                <div className="flex flex-wrap gap-2">
                  {(['share', 'off'] as const).map(choice => (
                    <Button disabled={busy} key={choice} onClick={() => void chooseConsent(choice)} variant="outline">
                      {t.sharedMetrics.stripChoices[choice]}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="grid gap-4">
            <h2 className="font-medium">{copy.model}</h2>
            <ModelReadiness checking={checking} ready={ready} runtime={runtime} />
            {failure && (
              <p className="text-sm" role="alert">
                {setupFailureCopy(failure, t.freeTier.setupFailed)}
              </p>
            )}
            {!ready && free?.enabled === false && (
              <p className="text-sm text-muted-foreground">{copy.freeUnavailable}</p>
            )}
            <div className="flex flex-wrap gap-2">
              {!ready && free?.enabled && (
                <Button
                  disabled={busy || checking || (failure !== null && !failure.retryable)}
                  onClick={() => void provision()}
                >
                  {copy.freeModel}
                </Button>
              )}
              <Button disabled={busy} onClick={() => startManualOnboarding(null, scope)} variant="outline">
                {copy.providers}
              </Button>
              <Button disabled={busy || checking} onClick={() => void check()} variant="ghost">
                {copy.check}
              </Button>
            </div>
          </div>
        )}
        {error && (
          <p className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Button disabled={busy} onClick={() => void act(() => finishOnboarding('skipped'))} variant="ghost">
            {copy.skip}
          </Button>
          <div className="flex flex-wrap gap-2">
            {step === 'model' && (
              <Button disabled={busy} onClick={() => void act(() => saveGuideStep('preferences'))} variant="outline">
                {copy.back}
              </Button>
            )}
            {step === 'preferences' ? (
              <Button disabled={busy} onClick={() => void act(() => saveGuideStep('model'))}>
                {t.common.continue}
              </Button>
            ) : (
              <Button disabled={busy || checking || !ready} onClick={() => void finish()}>
                {copy.finish}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

interface ModelReadinessProps {
  checking: boolean
  ready: boolean
  runtime: RuntimeReadinessResult | null
}

function ModelReadiness({ checking, ready, runtime }: ModelReadinessProps) {
  const { t } = useI18n()
  const copy = t.firstRun.guide

  return (
    <>
      <p role="status">{checking ? copy.checking : ready ? copy.ready : runtime?.reason || copy.providerNeeded}</p>
      {ready && runtime?.model && <p className="text-sm text-muted-foreground">{runtime.model}</p>}
    </>
  )
}
