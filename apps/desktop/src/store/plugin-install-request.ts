import { atom } from 'nanostores'

/** Which plugin component(s) a legacy deeplink pre-selects after probe. */
export type PluginInstallLegacyHint = 'agent' | 'desktop' | null

/** Future metrics (opt-in): install count + success/failure, per repo. */
export interface PluginInstallRequest {
  repo: string
  enable?: boolean
  force?: boolean
  legacyHint?: PluginInstallLegacyHint
}

export const $pluginInstallRequest = atom<PluginInstallRequest | null>(null)

export function openPluginInstallRequest(request: PluginInstallRequest): void {
  $pluginInstallRequest.set(request)
}

export function closePluginInstallRequest(): void {
  $pluginInstallRequest.set(null)
}
