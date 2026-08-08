import { useCallback, useEffect, useRef, useState } from 'react'

import { pollOAuthSession, startOAuthLogin } from '@/hermes'
import { notify, notifyError } from '@/store/notifications'

interface UseNousPortalLoginOptions {
  failureMessage: string
  onApproved?: () => void
  successMessage: string
  successTitle: string
}

async function openSignInUrl(url: string) {
  if (window.hermesDesktop?.openExternal) {
    try {
      await window.hermesDesktop.openExternal(url)

      return
    } catch {
      // Keep a broken native bridge from stranding an otherwise valid OAuth session.
    }
  }

  window.open(url, '_blank', 'noopener,noreferrer')
}

/**
 * Starts and observes the shared Nous Portal device-code flow for settings
 * surfaces. The backend owns the credential; callers only refresh their own
 * cached view after approval.
 */
export function useNousPortalLogin({
  failureMessage,
  onApproved,
  successMessage,
  successTitle
}: UseNousPortalLoginOptions) {
  const mountedRef = useRef(true)
  const inFlightRef = useRef(false)
  const [signingIn, setSigningIn] = useState(false)

  // eslint-disable-next-line no-restricted-syntax -- mount flag guards an async poll loop, not an atom mirror.
  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
    }
  }, [])

  const signInToNousPortal = useCallback(async () => {
    if (inFlightRef.current) {
      return
    }

    inFlightRef.current = true
    setSigningIn(true)

    try {
      const start = await startOAuthLogin('nous')

      if (start.flow !== 'device_code') {
        notifyError(new Error(`unexpected flow: ${start.flow}`), failureMessage)

        return
      }

      await openSignInUrl(start.verification_url)

      for (let attempt = 0; attempt < 120 && mountedRef.current; attempt += 1) {
        await new Promise(resolve => window.setTimeout(resolve, 5000))

        if (!mountedRef.current) {
          return
        }

        const polled = await pollOAuthSession('nous', start.session_id)

        if (polled.status === 'approved') {
          notify({ kind: 'success', title: successTitle, message: successMessage })
          onApproved?.()

          return
        }

        if (polled.status !== 'pending') {
          notifyError(new Error(polled.error_message || `Sign-in ${polled.status}`), failureMessage)

          return
        }
      }
    } catch (err) {
      if (mountedRef.current) {
        notifyError(err, failureMessage)
      }
    } finally {
      inFlightRef.current = false

      if (mountedRef.current) {
        setSigningIn(false)
      }
    }
  }, [failureMessage, onApproved, successMessage, successTitle])

  return { signInToNousPortal, signingIn }
}
