import { useRef, useState } from 'react'

import type { DesktopConnectionConfigInput, DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import { deriveRemoteAuthProviderShape } from '@/lib/desktop-remote-auth'
import { coerceRemoteUrlScheme } from '@/lib/remote-url'
import type { NotificationInput } from '@/store/notifications'

import { type RemoteConnectionTest, useRemoteConnectionTest } from './use-remote-connection-test'
import { type RemoteOAuth, useRemoteOAuth } from './use-remote-oauth'
import { useRemoteProbe } from './use-remote-probe'

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
  /**
   * Registry-draft identity for a sign-in that runs before the draft is
   * saved; see useRemoteOAuth. Only the registry host supplies it.
   */
  oauthLoginIdentity?: () => { connectionId: null | string; label: string }
  /** Reports the settled connection id a pre-save sign-in wrote the session for. */
  onOAuthLoginSettled?: (connectionId: string) => void
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
  const { host, enabled = true } = options
  const callbacks = useRef<RemoteSetupOptions>(options)
  callbacks.current = options
  const [credentials, setCredentials] = useState<RemoteCredentials>(credentialsFrom)
  const [revision, setRevision] = useState<number>(0)

  const url = coerceRemoteUrlScheme(credentials.url)
  const manualAuth = host === 'registry'
  const probeEnabled = enabled && (!manualAuth || credentials.authMode === 'oauth')
  const notify = (notice: NotificationInput): void => callbacks.current.onNotice?.(notice)

  const { probe, probeStatus, targetSeq, invalidateProbe, acceptProbe } = useRemoteProbe({
    enabled: probeEnabled,
    url,
    revision,
    onReset: (): void => {
      oauth.clearSigningIn()
      connectionTest.clearTestOutcome()
    },
    onResult: (result: DesktopConnectionProbeResult): void => {
      connectionTest.invalidateTest()

      if (!manualAuth && result.reachable && result.authMode !== 'unknown') {
        const authMode = result.authMode
        setCredentials(current => ({
          ...current,
          authMode,
          oauthConnected: authMode === 'oauth' && current.oauthConnected
        }))
      }
    }
  })

  const payload: DesktopConnectionConfigInput = {
    mode: 'remote',
    remoteAuthMode: credentials.authMode,
    remoteToken: credentials.authMode === 'token' ? credentials.token.trim() || undefined : undefined,
    remoteUrl: url
  }

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

  const connectionTest: RemoteConnectionTest = useRemoteConnectionTest({
    host,
    url,
    payload,
    canTest,
    authResolved,
    targetSeq,
    acceptProbe,
    notify
  })

  const oauth: RemoteOAuth = useRemoteOAuth({
    host,
    url,
    providerLabel,
    targetSeq,
    beforeOAuthLogin: (value: DesktopConnectionConfigInput): Promise<void> | undefined =>
      callbacks.current.beforeOAuthLogin?.(value),
    oauthLoginIdentity: (): { connectionId: null | string; label: string } | undefined =>
      callbacks.current.oauthLoginIdentity?.(),
    onOAuthLoginSettled: (connectionId: string): void => {
      callbacks.current.onOAuthLoginSettled?.(connectionId)
    },
    setOAuthConnected: (oauthConnected: boolean): void => setCredentials(value => ({ ...value, oauthConnected })),
    invalidateTest: connectionTest.invalidateTest,
    reportError: connectionTest.reportError,
    notify
  })

  const invalidateTarget = (): void => {
    invalidateProbe()
    oauth.invalidateLogin()
    connectionTest.invalidateTest()
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
    connectionTest.invalidateTest()
    setCredentials(current => ({ ...current, token }))
  }

  const setAuthMode = (authMode: AuthMode): void => {
    invalidateTarget()
    setCredentials(current => ({ ...current, authMode, oauthConnected: false }))
    setRevision(value => value + 1)
  }

  return {
    host,
    credentials,
    payload,
    probeStatus,
    authResolved,
    providerLabel,
    isPassword,
    signingIn: oauth.signingIn,
    testing: connectionTest.testing,
    error: connectionTest.error,
    success: connectionTest.success,
    canTest,
    canCommit: enabled && (host === 'first-run' ? connectionTest.passed : host === 'registry' || credentialReady),
    setUrl,
    setToken,
    setAuthMode,
    reset,
    signIn: oauth.signIn,
    signOut: oauth.signOut,
    test: connectionTest.test
  }
}
