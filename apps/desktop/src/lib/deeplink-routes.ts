import type { PluginInstallLegacyHint } from '@/store/plugin-install-request'

export interface DeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

export type DeepLinkAction =
  | { type: 'plugin-install'; repo: string; enable: boolean; force: boolean; legacyHint: PluginInstallLegacyHint }
  /** `hermes://plugin/install?catalog=<name>` — resolved against the curated
   *  catalog by the caller; the raw name is never treated as a git identifier. */
  | { type: 'plugin-catalog-install'; name: string }
  | { type: 'composer-blueprint'; name: string; params: Record<string, string> }
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

  // A `catalog` param claims the link outright: even when a `repo` rides along
  // (or the name is empty/bogus) the outcome is the catalog lookup's verdict,
  // never a git-path install of whatever else the link carried.
  if (payload.kind === 'plugin' && payload.name === 'install' && payload.params?.catalog !== undefined) {
    return { type: 'plugin-catalog-install', name: payload.params.catalog.trim() }
  }

  const repo = (payload.params?.repo || payload.params?.identifier || payload.name || '').trim()

  if (payload.kind === 'plugin' && payload.name === 'install' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: null
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
