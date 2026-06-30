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

// A remote, gated (oauth-bucket), not-currently-connected gateway is a
// remote-reauth boot failure: the access cookie lapsed (e.g. the remote
// dashboard restarted) and the local-recovery buttons (Retry/Repair) can't
// fix it — only re-establishing the remote session can. A connected oauth
// session, or a token/local gateway, boots for some other reason the
// local-recovery buttons address, so those return false here.
function isRemoteAuthError(message?: string | null): boolean {
  return /sign in|session (?:has )?expired|not signed in|oauth session|websocket ticket/i.test(String(message || ''))
}

export function isRemoteReauthFailure(
  config: DesktopConnectionConfig | null | undefined,
  errorMessage?: string | null
): boolean {
  if (!config) {
    return false
  }

  return (
    config.mode === 'remote' &&
    config.remoteAuthMode === 'oauth' &&
    (!config.remoteOauthConnected || isRemoteAuthError(errorMessage)) &&
    Boolean(config.remoteUrl)
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
