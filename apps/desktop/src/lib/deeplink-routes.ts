import type { PluginInstallLegacyHint } from '@/store/plugin-install-request'

export interface DeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

export type DeepLinkAction =
  | {
      type: 'plugin-install'
      repo: string
      enable: boolean
      force: boolean
      legacyHint: PluginInstallLegacyHint
      catalogName?: string
      sha?: string
    }
  | { type: 'skill-install'; identifier: string }
  | { type: 'composer-blueprint'; name: string; params: Record<string, string> }
  | { type: 'bot-open'; profile: string }
  | { type: 'ignore' }

function truthyParam(value: string | undefined, defaultValue = false): boolean {
  if (value === undefined || value === '') {
    return defaultValue
  }

  const normalized = value.trim().toLowerCase()

  return normalized === '1' || normalized === 'true' || normalized === 'yes'
}

export function resolveDeepLinkAction(payload: DeepLinkPayload | null | undefined): DeepLinkAction {
  if (!payload?.kind) {
    return { type: 'ignore' }
  }

  if (payload.kind === 'blueprint' && payload.name) {
    return { type: 'composer-blueprint', name: payload.name, params: payload.params || {} }
  }

  if (payload.kind === 'skill') {
    const identifier = payload.params?.identifier

    return payload.name === 'install' && identifier && identifier === identifier.trim()
      ? { type: 'skill-install', identifier }
      : { type: 'ignore' }
  }

  // hermes://bot/<profile> (or hermes://open/bots?bot=<profile>): open that
  // bot's canonical Bot Chat. The link carries a NAME, never a session id —
  // the consumer resolves the registry at open time, so the link can't dangle.
  const botProfile = payload.kind === 'bot' ? payload.name : payload.params?.bot || ''

  if (payload.kind === 'bot' || (payload.kind === 'open' && payload.name === 'bots')) {
    const profile = botProfile.trim()

    return profile ? { type: 'bot-open', profile } : { type: 'ignore' }
  }

  const repo = (
    payload.params?.repo || payload.params?.identifier || (payload.kind !== 'plugin' ? payload.name : '') || ''
  ).trim()

  if (payload.kind === 'plugin' && payload.name === 'install' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: null,
      catalogName: payload.params?.catalog_name || undefined,
      sha: payload.params?.sha || undefined
    }
  }

  if (payload.kind === 'plugin-agent' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: 'agent'
    }
  }

  if (payload.kind === 'plugin-desktop' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: 'desktop'
    }
  }

  return { type: 'ignore' }
}
