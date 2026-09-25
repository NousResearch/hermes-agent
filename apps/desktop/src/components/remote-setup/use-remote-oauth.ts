import { type RefObject, useRef, useState } from 'react'

import type { DesktopConnectionConfigInput } from '@/global'
import { useI18n } from '@/i18n'
import type { NotificationInput } from '@/store/notifications'

import type { RemoteSetupHost } from './use-remote-setup'

interface RemoteOAuthOptions {
  host: RemoteSetupHost
  url: string
  providerLabel: string
  targetSeq: RefObject<number>
  beforeOAuthLogin: (payload: DesktopConnectionConfigInput) => Promise<void> | undefined
  setOAuthConnected: (connected: boolean) => void
  invalidateTest: () => void
  reportError: (err: unknown, title?: string, kind?: 'error' | 'warning') => void
  notify: (notice: NotificationInput) => void
}

export interface RemoteOAuth {
  signingIn: boolean
  // Retargeting orphans any in-flight login; its result must not land.
  invalidateLogin: () => void
  clearSigningIn: () => void
  signIn: () => Promise<void>
  signOut: () => Promise<void>
}

/** OAuth leg of the remote editor: login/logout generations fenced by the probe target. */
export function useRemoteOAuth(options: RemoteOAuthOptions): RemoteOAuth {
  const { t } = useI18n()
  const g = t.settings.gateway

  const {
    host,
    url,
    providerLabel,
    targetSeq,
    beforeOAuthLogin,
    setOAuthConnected,
    invalidateTest,
    reportError,
    notify
  } = options

  const [signingIn, setSigningIn] = useState<boolean>(false)
  const loginSeq = useRef<number>(0)

  const invalidateLogin = (): void => {
    loginSeq.current += 1
    setSigningIn(false)
  }

  const clearSigningIn = (): void => {
    setSigningIn(false)
  }

  const signIn = async (): Promise<void> => {
    if (!url || signingIn) {
      return
    }

    const target: number = targetSeq.current
    const seq: number = ++loginSeq.current
    const current = (): boolean => target === targetSeq.current && seq === loginSeq.current
    invalidateTest()
    setSigningIn(true)

    try {
      await beforeOAuthLogin({ mode: 'remote', remoteAuthMode: 'oauth', remoteUrl: url })

      if (!current()) {
        return
      }

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(url)

      if (!current()) {
        return
      }

      setOAuthConnected(Boolean(result.connected))

      if (result.connected) {
        notify({ kind: 'success', title: g.signedIn, message: g.connectedTo(providerLabel) })
      } else {
        const message = host === 'first-run' ? t.install.signInIncomplete : t.boot.failure.signInIncompleteMessage
        reportError(
          result.error ? `${message}: ${result.error}` : message,
          t.boot.failure.signInIncompleteTitle,
          'warning'
        )
      }
    } catch (err) {
      if (current()) {
        reportError(err, g.signInFailed)
      }
    } finally {
      if (current()) {
        setSigningIn(false)
      }
    }
  }

  const signOut = async (): Promise<void> => {
    const target: number = targetSeq.current
    const seq: number = ++loginSeq.current
    const current = (): boolean => target === targetSeq.current && seq === loginSeq.current
    invalidateTest()
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(url)

      if (current()) {
        setOAuthConnected(false)
        notify({ kind: 'success', title: g.signedOutTitle, message: g.signedOutMessage })
      }
    } catch (err) {
      if (current()) {
        reportError(err, g.signOutFailed)
      }
    } finally {
      if (current()) {
        setSigningIn(false)
      }
    }
  }

  return { signingIn, invalidateLogin, clearSigningIn, signIn, signOut }
}
