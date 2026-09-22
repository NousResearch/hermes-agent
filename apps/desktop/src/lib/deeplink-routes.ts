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
  /** `hermes://bot/install?catalog=<name>` — the catalog key is the only
   * accepted input; every other query field is intentionally discarded. */
  | { type: 'bot-catalog-install'; name: string }
  | { type: 'composer-blueprint'; name: string; params: Record<string, string> }
  | { type: 'connection-done'; op: string; status: string }
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

  // The browser leg of a connection came back (hermes://connections/done?op=…&status=…). The op id
  // names the operation to show; the status is carried but never moves a row, because the link is
  // whatever the user's browser was pointed at.
  if (payload.kind === 'connections' && payload.name === 'done') {
    const op = (payload.params?.op || '').trim()

    return op ? { type: 'connection-done', op, status: (payload.params?.status || '').trim() } : { type: 'ignore' }
  }

  if (payload.kind === 'bot' && payload.name === 'install') {
    const catalog = (payload.params?.catalog || '').trim()

    return catalog ? { type: 'bot-catalog-install', name: catalog } : { type: 'ignore' }
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
