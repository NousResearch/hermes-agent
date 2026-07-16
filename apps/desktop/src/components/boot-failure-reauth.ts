import type { DesktopAuthProvider, DesktopConnectionConfig } from '@/global'

// Pure helpers for the boot-failure overlay's remote-reauth branch. Kept out
// of the .tsx so they can be unit-tested without a React/jsdom render (the
// jsx-dev-runtime resolution in this repo's vitest setup is flaky for
// component renders, but these are plain functions).

export interface RemoteReauth {
  url: string
  // True when every advertised provider is username/password — drives the
  // button copy ("Sign in to remote gateway" vs "Sign in with <provider>"),
  // mirroring the gateway-settings page. Probe is best-effort.
  isPassword: boolean
  providerLabel: string
}

interface SignInCopy {
  identityProvider: string
  remoteGateway: string
  withProvider: (provider: string) => string
}

const DEFAULT_SIGN_IN_COPY: SignInCopy = {
  identityProvider: 'your identity provider',
  remoteGateway: 'Sign in to remote gateway',
  withProvider: provider => `Sign in with ${provider}`
}

// True when the app is pointed at a remote/cloud backend (either resolves to a
// remote URL). Any boot failure in this shape is fixable from Settings →
// Gateway (edit URL / token / sign in) — the local Retry/Repair buttons target
// the bundled backend and can't help. Drives the escape-hatch emphasis.
export function isRemoteConfig(config: DesktopConnectionConfig | null | undefined): boolean {
  return Boolean(config && (config.mode === 'remote' || config.mode === 'cloud') && config.remoteUrl)
}

// True when a boot error is auth-shaped — the refresh token was rejected or the
// remote couldn't mint a websocket ticket. The Settings indicator can still read
// "connected" (a stale RT cookie exists), so the error text is part of the
// signal; without it a connected-but-expired session drops into the local-only
// recovery buttons for a problem only reauth can fix.
export function isRemoteReauthError(error: string | null | undefined): boolean {
  const text = String(error || '').toLowerCase()

  return (
    text.includes('remote gateway session has expired') ||
    text.includes('gateway sign-in required') ||
    text.includes('needs oauth login') ||
    (text.includes('oauth') && (text.includes('not signed in') || text.includes('sign in')))
  )
}

// A remote, gated (oauth-bucket) gateway is a remote-reauth boot failure when the
// session isn't connected OR the boot error is auth-shaped (connected-but-expired
// — see isRemoteReauthError). Only re-establishing the remote session fixes it;
// the local Retry/Repair buttons can't. 'cloud' counts as remote (it resolves to
// a remote oauth backend), so a lapsed cloud session is the same failure.
export function isRemoteReauthFailure(
  config: DesktopConnectionConfig | null | undefined,
  error?: string | null
): boolean {
  return (
    isRemoteConfig(config) &&
    config!.remoteAuthMode === 'oauth' &&
    (!config!.remoteOauthConnected || isRemoteReauthError(error))
  )
}

// Derive the password flag + display label from the probed providers. A
// gateway is treated as password-style only when EVERY advertised provider
// supports password (a mixed deployment keeps the generic OAuth copy), so the
// button copy matches the login window the user is about to see.
export function deriveProviderShape(providers: DesktopAuthProvider[] | null | undefined): {
  isPassword: boolean
  providerLabel: string
} {
  const list = providers ?? []

  if (list.length === 0) {
    return { isPassword: false, providerLabel: 'your identity provider' }
  }

  const isPassword = list.every(p => Boolean(p.supportsPassword))

  const providerLabel =
    list.length === 1 ? list[0].displayName || list[0].name : list.map(p => p.displayName || p.name).join(' / ')

  return { isPassword, providerLabel }
}

// Button copy for the remote sign-in action.
export function signInLabel(reauth: RemoteReauth | null, copy: SignInCopy = DEFAULT_SIGN_IN_COPY): string {
  if (reauth?.isPassword) {
    return copy.remoteGateway
  }

  const provider =
    reauth?.providerLabel === DEFAULT_SIGN_IN_COPY.identityProvider ? copy.identityProvider : reauth?.providerLabel

  return copy.withProvider(provider ?? copy.identityProvider)
}
