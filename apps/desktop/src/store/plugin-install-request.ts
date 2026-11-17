import { atom } from 'nanostores'

import type { ProfileScope } from '@/hermes'

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
  /** Catalog tier (official / verified / community), display only: an
   *  official pick is badged Official rather than Verified. */
  tier?: string
  /** The catalog pin (display only — the backend resolves it itself). */
  sha?: string
  /** Display name for the dialog heading (the catalog card's title); falls back to the catalog name. */
  title?: string
  /** One line on what the plugin does (the catalog description's first sentence). */
  summary?: string
  /** Exact Capabilities owner; legacy strings/null retain their ambient route. */
  profile?: ProfileScope
  /** The caller keeps the user where they are after a successful install (the setup chat's plugins card). */
  stayAfterInstall?: boolean
}

export const $pluginInstallRequest = atom<PluginInstallRequest | null>(null)

/** One explicit user-initiated help draft, delivered only after fresh-chat rotation. */
export const $pendingPluginHelpDraft = atom<string | null>(null)

export function queuePluginHelpDraft(text: string): void {
  $pendingPluginHelpDraft.set(text)
}

export function takePluginHelpDraft(): string | null {
  const text = $pendingPluginHelpDraft.get()
  $pendingPluginHelpDraft.set(null)

  return text
}

export function openPluginInstallRequest(request: PluginInstallRequest): void {
  $pluginInstallRequest.set(request)
}

export function closePluginInstallRequest(): void {
  $pluginInstallRequest.set(null)
}
