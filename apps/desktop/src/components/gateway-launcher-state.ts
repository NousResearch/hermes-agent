import type { DesktopConnectionConfig } from '@/global'

import { isRemoteReauthFailure } from './boot-failure-reauth'

export type GatewayStartupKind = 'setup' | 'signin' | 'unavailable' | 'unexpected'

export interface GatewayStartupState {
  config: DesktopConnectionConfig | null
  diagnostic: string
  kind: GatewayStartupKind
  remoteUrl: string
}

export function hasUsableRemoteUrl(config: DesktopConnectionConfig | null | undefined): boolean {
  return Boolean(config?.remoteUrl?.trim())
}

export function isUnexpectedGatewayError(message?: string | null): boolean {
  const text = String(message || '')

  return /expected json|invalid json|got html|endpoint is likely missing|unexpected token|404:|405:|not found/i.test(text)
}

export function isNetworkGatewayError(message?: string | null): boolean {
  const text = String(message || '')

  return (
    /did not become reachable|could not reach|timed out|timeout|econnrefused|enotfound|eai_again|etimedout|econnreset|socket hang up|getaddrinfo|connection refused|network/i.test(
      text
    ) || !isUnexpectedGatewayError(text)
  )
}

export function summarizeGatewayDiagnostic(message?: string | null): string {
  const text = String(message || '').trim()

  if (!text) {
    return 'No diagnostic was reported.'
  }

  if (/econnrefused|connection refused/i.test(text)) {
    return 'Connection refused. The host is reachable, but nothing accepted the gateway connection on that port.'
  }

  if (/enotfound|getaddrinfo|eai_again/i.test(text)) {
    return 'DNS lookup failed. Reuben could not resolve the configured gateway host.'
  }

  if (/timed out|timeout|etimedout/i.test(text)) {
    return 'Connection timed out. The configured gateway did not respond within the startup window.'
  }

  if (isUnexpectedGatewayError(text)) {
    return 'The host responded, but it did not look like a compatible Reuben/Hermes gateway.'
  }

  return text.replace(/^Error:\s*/, '')
}

export function classifyGatewayStartup(
  config: DesktopConnectionConfig | null | undefined,
  errorMessage?: string | null
): GatewayStartupState {
  const remoteUrl = config?.remoteUrl?.trim() ?? ''
  const diagnostic = summarizeGatewayDiagnostic(errorMessage)

  if (!hasUsableRemoteUrl(config)) {
    return { config: config ?? null, diagnostic, kind: 'setup', remoteUrl: '' }
  }

  if (isRemoteReauthFailure(config, errorMessage)) {
    return { config: config ?? null, diagnostic, kind: 'signin', remoteUrl }
  }

  if (isUnexpectedGatewayError(errorMessage)) {
    return { config: config ?? null, diagnostic, kind: 'unexpected', remoteUrl }
  }

  if (isNetworkGatewayError(errorMessage)) {
    return { config: config ?? null, diagnostic, kind: 'unavailable', remoteUrl }
  }

  return { config: config ?? null, diagnostic, kind: 'unexpected', remoteUrl }
}
