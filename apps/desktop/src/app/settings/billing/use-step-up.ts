import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
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

export function useStepUpFlow() {
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
    unsubscribe()
    setMessage(null)
    setPhase('idle')
    setVerification(null)
  }, [unsubscribe])

  useEffect(
    () => () => {
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

  const start = useCallback(async () => {
    if (runningRef.current) {
      return
    }

    runningRef.current = true
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
      }) ?? null

    const result = await api.stepUp()

    if (runIdRef.current !== runId) {
      return
    }

    runningRef.current = false
    unsubscribe()

    if (!result.ok) {
      setMessage({ status: 'refusal', refusal: result.refusal })

      return
    }

    if (!result.data.granted) {
      setMessage({ status: 'denied' })

      return
    }

    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['billing', 'state'] }),
      queryClient.invalidateQueries({ queryKey: ['billing', 'subscription'] })
    ])
    setMessage({ status: 'success' })
  }, [api, gateway, queryClient, unsubscribe])

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
