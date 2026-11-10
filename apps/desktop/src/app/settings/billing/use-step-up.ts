import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { atom } from 'nanostores'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { $gateway } from '@/store/gateway'

import type { BillingRefusal } from './api'
import { useBillingApi } from './api'
import { resolveRefusal } from './errors'

export type StepUpPhase = 'idle' | 'verifying' | 'waiting'

export interface StepUpVerification {
  code: string | null
  url: string
}

export interface StepUpMessage {
  kind: 'error' | 'success'
  text: string
  title: string
}

/**
 * True while a Remote Spending approval is in flight. The device flow swaps this terminal's login when it
 * lands, so Billing pauses its focus/interval refetches meanwhile: a refetch on the old login mid-swap can
 * be refused, and a refused billing.state collapses the page (every money control, the row that started
 * the approval included, disappears). The flow's own completion refetches once on the new login.
 */
export const $stepUpInFlight = atom(false)

// Billing mounts several step-up flows (the Remote Spending card, refusal inlines, Keep going on credits).
// Each holds the pause under its own token, so one row dismissing its idle flow never resumes refetches
// while another row's approval is still in flight.
const stepUpHolders = new Set<object>()

function holdStepUp(holder: object, held: boolean) {
  if (held) {
    stepUpHolders.add(holder)
  } else {
    stepUpHolders.delete(holder)
  }

  $stepUpInFlight.set(stepUpHolders.size > 0)
}

export function useStepUpFlow({ autoOpen = false }: { autoOpen?: boolean } = {}) {
  const { t } = useI18n()
  const b = t.settings.billing
  const api = useBillingApi()
  const gateway = useStore($gateway)
  const queryClient = useQueryClient()
  const offRef = useRef<(() => void) | null>(null)
  const runningRef = useRef(false)
  const runIdRef = useRef(0)

  const [message, setMessage] = useState<
    null | { status: 'denied' | 'success' } | { status: 'refusal'; refusal: BillingRefusal }
  >(null)

  const [phase, setPhase] = useState<StepUpPhase>('idle')
  const [verification, setVerification] = useState<StepUpVerification | null>(null)

  const unsubscribe = useCallback(() => {
    offRef.current?.()
    offRef.current = null
  }, [])

  const dismiss = useCallback(() => {
    runIdRef.current += 1
    runningRef.current = false
    holdStepUp(runningRef, false)
    unsubscribe()
    setMessage(null)
    setPhase('idle')
    setVerification(null)
  }, [unsubscribe])

  useEffect(
    () => () => {
      holdStepUp(runningRef, false)
      runIdRef.current += 1
      runningRef.current = false
      unsubscribe()
    },
    [unsubscribe]
  )

  const openVerification = useCallback(() => {
    if (!verification?.url) {
      return
    }

    void window.hermesDesktop?.openExternal?.(verification.url)
  }, [verification?.url])

  /** Resolves true once Remote Spending is allowed (billing queries already refetched), else false. */
  const start = useCallback(async (): Promise<boolean> => {
    if (runningRef.current) {
      return false
    }

    runningRef.current = true
    holdStepUp(runningRef, true)
    const runId = runIdRef.current + 1

    runIdRef.current = runId
    unsubscribe()
    setMessage(null)
    setVerification(null)
    setPhase('waiting')

    offRef.current =
      gateway?.on('billing.step_up.verification', event => {
        const payload = event.payload
        const url = typeof payload?.verification_url === 'string' ? payload.verification_url : null

        if (!url) {
          return
        }

        setVerification({
          code: typeof payload?.user_code === 'string' ? payload.user_code : null,
          url
        })
        setPhase('verifying')

        // The user just asked for this (clicked a control that needs it): open the approval page
        // right away, like sign-in does, instead of making them click a second button for it.
        if (autoOpen) {
          void window.hermesDesktop?.openExternal?.(url)
        }
      }) ?? null

    const result = await api.stepUp()

    if (runIdRef.current !== runId) {
      return false
    }

    runningRef.current = false
    holdStepUp(runningRef, false)
    unsubscribe()
    setPhase('idle')
    setVerification(null)

    if (!result.ok) {
      setMessage({ status: 'refusal', refusal: result.refusal })

      return false
    }

    if (!result.data.granted) {
      setMessage({ status: 'denied' })

      return false
    }

    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['billing', 'state'] }),
      queryClient.invalidateQueries({ queryKey: ['billing', 'subscription'] })
    ])
    setMessage({ status: 'success' })

    return true
  }, [api, autoOpen, gateway, queryClient, unsubscribe])

  let displayMessage: StepUpMessage | null = null

  if (message?.status === 'refusal') {
    const resolved = resolveRefusal(message.refusal, b.errors)
    displayMessage = { kind: 'error', text: resolved.message, title: resolved.title }
  } else if (message) {
    displayMessage =
      message.status === 'success'
        ? { kind: 'success', text: b.stepUp.successBody, title: b.stepUp.successTitle }
        : { kind: 'error', text: b.stepUp.deniedBody, title: b.stepUp.deniedTitle }
  }

  return { dismiss, message: displayMessage, openVerification, phase, start, verification }
}
