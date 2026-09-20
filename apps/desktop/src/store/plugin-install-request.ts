import { atom } from 'nanostores'

import { captureOwner, type OwnerScope, type ProfileScope } from '@/api/client'

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
  /** Scope the pick was made under; resolved to an owner when the request opens, not when it completes. */
  profile?: ProfileScope
  /** Where the install was started from, so a finished install can return there. */
  origin?: { kind: 'memory'; providerId: string }
}

export interface CapturedPluginInstallRequest extends Omit<PluginInstallRequest, 'profile'> {
  target: OwnerScope
}

export const $pluginInstallRequest = atom<CapturedPluginInstallRequest | null>(null)

export function openPluginInstallRequest({ profile, ...request }: PluginInstallRequest): void {
  $pluginInstallRequest.set({ ...request, target: captureOwner(profile) })
}

export function closePluginInstallRequest(): void {
  $pluginInstallRequest.set(null)
}
