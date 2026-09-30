import { type RefObject, useRef, useState } from 'react'

import type { DesktopConnectionConfigInput, DesktopOauthLoginOptions } from '@/global'
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
  /**
   * Registry-draft identity for a sign-in that runs BEFORE the draft is
   * saved. The main process derives the login window's cookie partition from
   * the settled connection id, gated on the draft's kind/authMode — only a
   * cookie-auth remote draft gets its own jar; cloud and token drafts sign in
   * on the legacy shared jar the saved entry reads. Without the identity an
   * unsaved draft's session lands in the legacy shared jar the saved
   * connection never reads. Absent on the first-run/settings hosts, which
   * have no draft identity.
   */
  oauthLoginIdentity?: () => DesktopOauthLoginOptions | undefined
  /** Reports the settled id a pre-save sign-in wrote the session for. */
  onOAuthLoginSettled?: (connectionId: string) => void
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
    notify,
    oauthLoginIdentity,
    onOAuthLoginSettled
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

      // Absent identity (first-run/settings hosts) keeps the legacy single-arg
      // call; the registry host always supplies one for its draft.
      const identity = oauthLoginIdentity?.()

      const result = identity
        ? await window.hermesDesktop.oauthLoginConnectionConfig(url, identity)
        : await window.hermesDesktop.oauthLoginConnectionConfig(url)

      if (!current()) {
        return
      }

      // A pre-save sign-in settles the id the later save will reuse; the
      // registry host pins it into the draft so both agree on the jar.
      if (result.connectionId) {
        onOAuthLoginSettled?.(result.connectionId)
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
    if (!url) {
      return
    }

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
