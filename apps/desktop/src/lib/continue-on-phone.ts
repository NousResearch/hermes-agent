import type { DesktopConnectionProbeResult } from '@/global'
import { getDashboardRemoteAccess } from '@/hermes'

export type ContinueOnPhoneFailureReason =
  | 'auth-required'
  | 'insecure-url'
  | 'not-configured'
  | 'unreachable'

export type ContinueOnPhoneResult =
  | { ok: true; url: string }
  | { ok: false; reason: ContinueOnPhoneFailureReason }

export interface ContinueOnPhoneDependencies {
  getRemoteAccess: (profile?: string) => ReturnType<typeof getDashboardRemoteAccess>
  probe: (publicUrl: string) => Promise<Pick<DesktopConnectionProbeResult, 'authMode' | 'reachable'>>
}

const DEFAULT_DEPENDENCIES: ContinueOnPhoneDependencies = {
  getRemoteAccess: getDashboardRemoteAccess,
  probe: publicUrl => window.hermesDesktop.probeConnectionConfig(publicUrl)
}

export function buildDashboardSessionUrl(
  publicUrl: string,
  sessionId: string,
  profile?: string
): string | null {
  const cleanSessionId = sessionId.trim()

  if (!cleanSessionId) {
    return null
  }

  let url: URL

  try {
    url = new URL(publicUrl)
  } catch {
    return null
  }

  if (url.protocol !== 'https:' || url.username || url.password) {
    return null
  }

  url.pathname = `${url.pathname.replace(/\/+$/, '')}/chat`
  url.search = ''
  url.hash = ''
  url.searchParams.set('resume', cleanSessionId)

  if (profile?.trim()) {
    url.searchParams.set('profile', profile.trim())
  }

  return url.toString()
}

export async function resolveContinueOnPhoneUrl(
  sessionId: string,
  profile?: string,
  dependencies: ContinueOnPhoneDependencies = DEFAULT_DEPENDENCIES
): Promise<ContinueOnPhoneResult> {
  const { public_url: publicUrl } = await dependencies.getRemoteAccess(profile)

  if (!publicUrl) {
    return { ok: false, reason: 'not-configured' }
  }

  const url = buildDashboardSessionUrl(publicUrl, sessionId, profile)

  if (!url) {
    return { ok: false, reason: 'insecure-url' }
  }

  const probe = await dependencies.probe(publicUrl)

  if (!probe.reachable) {
    return { ok: false, reason: 'unreachable' }
  }

  if (probe.authMode !== 'oauth') {
    return { ok: false, reason: 'auth-required' }
  }

  return { ok: true, url }
}
