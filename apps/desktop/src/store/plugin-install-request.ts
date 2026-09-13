import { atom } from 'nanostores'

/** Which plugin component(s) a legacy deeplink pre-selects after probe. */
export type PluginInstallLegacyHint = 'agent' | 'desktop' | null

/** Future metrics (opt-in): install count + success/failure, per repo. */
export interface PluginInstallRequest {
  repo: string
  enable?: boolean
  force?: boolean
  legacyHint?: PluginInstallLegacyHint
  /** Curated-catalog pick: install the agent half by catalog name so the
   *  backend pins the reviewed SHA and records sidecar provenance. */
  catalogName?: string
  /** The catalog pin (display only — the backend resolves it itself). */
  sha?: string
  /** Capabilities profile scope the pick was made under; the agent half
   *  installs into THIS profile (null/undefined = active profile). */
  profile?: string | null
}

export const $pluginInstallRequest = atom<PluginInstallRequest | null>(null)

export function openPluginInstallRequest(request: PluginInstallRequest): void {
  $pluginInstallRequest.set(request)
}

export function closePluginInstallRequest(): void {
  $pluginInstallRequest.set(null)
}
