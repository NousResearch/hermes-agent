import { useEffect, useRef, useState } from 'react'

import type { DesktopConnectionConfigInput, DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import { deriveRemoteAuthProviderShape } from '@/lib/desktop-remote-auth'
import { coerceRemoteUrlScheme } from '@/lib/remote-url'
import type { NotificationInput } from '@/store/notifications'

export type RemoteSetupHost = 'first-run' | 'settings' | 'registry'
type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'

export interface RemoteCredentials {
  url: string
  authMode: AuthMode
  token: string
  tokenSet: boolean
  tokenPreview: string | null
  oauthConnected: boolean
}

export interface RemoteSetupOptions {
  host: RemoteSetupHost
  enabled?: boolean
  beforeOAuthLogin?: (payload: DesktopConnectionConfigInput) => Promise<void>
  onNotice?: (notice: NotificationInput) => void
}

export interface RemoteSetup {
  host: RemoteSetupHost
  credentials: RemoteCredentials
  payload: DesktopConnectionConfigInput
  probeStatus: ProbeStatus
  authResolved: boolean
  providerLabel: string
  isPassword: boolean
  signingIn: boolean
  testing: boolean
  error: string | null
  success: string | null
  canTest: boolean
  canCommit: boolean
  setUrl: (url: string) => void
  setToken: (token: string) => void
  setAuthMode: (mode: AuthMode) => void
  reset: (saved?: Partial<RemoteCredentials>) => void
  signIn: () => Promise<void>
  signOut: () => Promise<void>
  test: () => Promise<void>
}

function credentialsFrom(saved: Partial<RemoteCredentials> = {}): RemoteCredentials {
  return { url: '', authMode: 'token', token: '', tokenSet: false, tokenPreview: null, oauthConnected: false, ...saved }
}

/**
 * Host contracts:
 * first-run: no pre-save; Apply requires a test of this exact payload.
 * settings: pre-save through beforeOAuthLogin; credentials permit Save/Apply.
 * registry: explicit auth selection; storage-only Save may precede credentials.
 * Persistence and live source changes belong to the host, never this editor.
 */
export function useRemoteSetup(options: RemoteSetupOptions): RemoteSetup {
  const { t } = useI18n()
  const g = t.settings.gateway
  const { host, enabled = true } = options
  const callbacks = useRef<RemoteSetupOptions>(options)
  callbacks.current = options
  const [credentials, setCredentials] = useState<RemoteCredentials>(credentialsFrom)
  const [revision, setRevision] = useState<number>(0)
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [signingIn, setSigningIn] = useState<boolean>(false)
  const [testing, setTesting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [testedKey, setTestedKey] = useState<string | null>(null)
  const targetSeq = useRef<number>(0)
  const testSeq = useRef<number>(0)
  const loginSeq = useRef<number>(0)
  const url = coerceRemoteUrlScheme(credentials.url)
  const manualAuth = host === 'registry'
  const probeEnabled = enabled && (!manualAuth || credentials.authMode === 'oauth')

  const payload: DesktopConnectionConfigInput = {
    mode: 'remote',
    remoteAuthMode: credentials.authMode,
    remoteToken: credentials.authMode === 'token' ? credentials.token.trim() || undefined : undefined,
    remoteUrl: url
  }

  const payloadKey = JSON.stringify(payload)
  const currentKey = useRef<string>(payloadKey)
  currentKey.current = payloadKey
  const { isPassword, providerLabel } = deriveRemoteAuthProviderShape(probe?.providers, t.boot.failure.identityProvider)

  const authResolved =
    manualAuth ||
    (probeStatus === 'done' && probe?.authMode !== 'unknown') ||
    (host === 'settings' && probeStatus === 'idle' && (credentials.tokenSet || credentials.oauthConnected))

  const credentialReady = Boolean(
    url &&
    (credentials.authMode === 'oauth' ? credentials.oauthConnected : credentials.token.trim() || credentials.tokenSet)
  )

  const canTest = enabled && Boolean(url) && ((authResolved && credentialReady) || probeStatus === 'error')

  const invalidateTest = (): void => {
    testSeq.current += 1
    setTesting(false)
    setTestedKey(null)
    setError(null)
    setSuccess(null)
  }

  const invalidateTarget = (): void => {
    targetSeq.current += 1
    loginSeq.current += 1
    setSigningIn(false)
    setProbe(null)
    setProbeStatus('idle')
    invalidateTest()
  }

  const reset = (saved?: Partial<RemoteCredentials>): void => {
    invalidateTarget()
    setCredentials(credentialsFrom(saved))
    setRevision(value => value + 1)
  }

  const setUrl = (value: string): void => {
    invalidateTarget()
    setCredentials(current => ({ ...current, url: value, oauthConnected: false, tokenSet: false, tokenPreview: null }))
    setRevision(value => value + 1)
  }

  const setToken = (token: string): void => {
    invalidateTest()
    setCredentials(current => ({ ...current, token }))
  }

  const setAuthMode = (authMode: AuthMode): void => {
    invalidateTarget()
    setCredentials(current => ({ ...current, authMode, oauthConnected: false }))
    setRevision(value => value + 1)
  }

  const reportError = (err: unknown, title: string = g.testFailed, kind: 'error' | 'warning' = 'error'): void => {
    const message = err instanceof Error ? err.message : String(err || g.testFailed)
    setError(message)
    callbacks.current.onNotice?.({ kind, title, message })
  }

  const reportSuccess = (message: string): void => {
    setSuccess(message)
    callbacks.current.onNotice?.({ kind: 'success', title: g.reachableTitle, message })
  }

  const acceptProbe = (result: DesktopConnectionProbeResult): void => {
    invalidateTest()
    setProbe(result)
    setProbeStatus(result.reachable ? 'done' : 'error')

    if (!manualAuth && result.reachable && result.authMode !== 'unknown') {
      const authMode = result.authMode
      setCredentials(current => ({
        ...current,
        authMode,
        oauthConnected: authMode === 'oauth' && current.oauthConnected
      }))
    }
  }

  // The effect reads current callbacks without restarting its debounce on each host render.
  const acceptProbeRef = useRef<(result: DesktopConnectionProbeResult) => void>(acceptProbe)
  acceptProbeRef.current = acceptProbe
  // eslint-disable-next-line no-restricted-syntax -- request generations, not a reactive value mirror
  useEffect(() => {
    const seq = ++targetSeq.current
    let timer: number | undefined

    const cancel = (): void => {
      targetSeq.current += 1
      window.clearTimeout(timer)
    }

    setProbe(null)
    setProbeStatus('idle')
    setSigningIn(false)
    setTesting(false)
    setTestedKey(null)
    setSuccess(null)

    if (!probeEnabled || !/^https?:\/\//i.test(url) || !window.hermesDesktop?.probeConnectionConfig) {
      return cancel
    }

    setProbeStatus('probing')
    timer = window.setTimeout(() => {
      void window.hermesDesktop
        .probeConnectionConfig(url)
        .then(result => {
          if (seq === targetSeq.current) {
            acceptProbeRef.current(result)
          }
        })
        .catch(() => {
          if (seq === targetSeq.current) {
            setProbeStatus('error')
          }
        })
    }, 500)

    return cancel
  }, [probeEnabled, revision, url])

  const signIn = async (): Promise<void> => {
    if (!url || signingIn) {
      return
    }

    const target = targetSeq.current
    const seq = ++loginSeq.current
    const current = (): boolean => target === targetSeq.current && seq === loginSeq.current
    invalidateTest()
    setSigningIn(true)

    try {
      await callbacks.current.beforeOAuthLogin?.({ mode: 'remote', remoteAuthMode: 'oauth', remoteUrl: url })

      if (!current()) {
        return
      }

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(url)

      if (!current()) {
        return
      }

      setCredentials(value => ({ ...value, oauthConnected: Boolean(result.connected) }))

      if (result.connected) {
        callbacks.current.onNotice?.({ kind: 'success', title: g.signedIn, message: g.connectedTo(providerLabel) })
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
    const target = targetSeq.current
    const seq = ++loginSeq.current
    const current = (): boolean => target === targetSeq.current && seq === loginSeq.current
    invalidateTest()
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(url)

      if (current()) {
        setCredentials(value => ({ ...value, oauthConnected: false }))
        callbacks.current.onNotice?.({ kind: 'success', title: g.signedOutTitle, message: g.signedOutMessage })
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

  const test = async (): Promise<void> => {
    if (!canTest) {
      return
    }

    const target = targetSeq.current
    const seq = ++testSeq.current

    const current = (): boolean =>
      target === targetSeq.current && seq === testSeq.current && payloadKey === currentKey.current

    setTesting(true)
    setError(null)
    setSuccess(null)
    setTestedKey(null)

    try {
      if (!authResolved) {
        const result = await window.hermesDesktop.probeConnectionConfig(url)

        if (current()) {
          acceptProbeRef.current(result)

          if (!result.reachable || result.authMode === 'unknown') {
            reportError(result.error || g.probeError)
          }
        }

        return
      }

      const result = await window.hermesDesktop.testConnectionConfig(payload)

      if (!current()) {
        return
      }

      if (result.ok === false || result.reachable === false) {
        throw new Error(result.error || g.testFailed)
      }

      reportSuccess(
        (host === 'first-run' ? t.install.testSucceeded : g.connectedTo)(
          result.baseUrl || url,
          result.version ?? undefined
        )
      )
      setTestedKey(payloadKey)
    } catch (err) {
      if (current()) {
        reportError(err)
      }
    } finally {
      if (current()) {
        setTesting(false)
      }
    }
  }

  return {
    host,
    credentials,
    payload,
    probeStatus,
    authResolved,
    providerLabel,
    isPassword,
    signingIn,
    testing,
    error,
    success,
    canTest,
    canCommit: enabled && (host === 'first-run' ? testedKey === payloadKey : host === 'registry' || credentialReady),
    setUrl,
    setToken,
    setAuthMode,
    reset,
    signIn,
    signOut,
    test
  }
}
