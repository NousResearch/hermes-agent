import { atom } from 'nanostores'

import type { ProfileScope } from '@/api/client'

/** Which plugin component(s) a legacy deeplink pre-selects after probe. */
export type PluginInstallLegacyHint = 'agent' | 'desktop' | null

/** Future metrics (opt-in): install count + success/failure, per repo. */
export interface PluginInstallRequest {
  /** Empty opens repository entry; a supplied repo goes straight to inspection. */
  repo: string
  enable?: boolean
  force?: boolean
  legacyHint?: PluginInstallLegacyHint
  /** Curated-catalog pick: install the agent half by catalog name so the
   *  backend pins the reviewed SHA and records sidecar provenance. */
  catalogName?: string
  /** The catalog pin (display only — the backend resolves it itself). */
  sha?: string
  /** Capabilities scope the pick was made under; the agent half installs into
   *  its profile (null/undefined = active profile) and a finished install
   *  returns to it. */
  profile?: ProfileScope
  /** Where the install was started from, so a finished install can return there. */
  origin?: { kind: 'memory'; providerId: string }
}

/** Bare profile name of a request scope for the `plugins.manage` RPC. */
export function requestProfileName(scope: ProfileScope): null | string {
  return (typeof scope === 'string' ? scope : scope?.profile) || null
}

export const $pluginInstallRequest = atom<PluginInstallRequest | null>(null)

export function openPluginInstallRequest(request: PluginInstallRequest): void {
  $pluginInstallRequest.set(request)
}

export function closePluginInstallRequest(): void {
  $pluginInstallRequest.set(null)
}
