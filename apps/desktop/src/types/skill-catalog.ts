export type CatalogAction = 'install' | 'update' | 'uninstall'
export type ManagedState = 'not_installed' | 'installed' | 'removed' | 'unmanaged_local' | 'pending' | 'unavailable'

export interface CatalogProvider {
  id: string
  plugin_id: string
  label: string
  version: string | null
  author: string | null
  managed_state: ManagedState
  scan_verdict: 'safe' | 'caution' | 'dangerous' | 'unknown'
  actions: CatalogAction[]
}

export interface CatalogSource {
  id: string
  label: string
  searchable: boolean
  provider: { id: string; plugin_id: string }
  state: 'ready' | 'hidden' | 'sign_in_required' | 'unavailable'
  message: string
}

export interface CatalogFile {
  path: string
  size: number
  change: 'add' | 'replace' | 'remove' | 'unchanged'
}

export interface CatalogFinding {
  id: string
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical'
  category: string
  file: string | null
  line: number | null
  description: string
  acknowledgement_required: boolean
}

export interface CatalogReview {
  review_id: string | null
  action: CatalogAction
  identifier: string
  version: string | null
  installation_revision: string | null
  target_relative_path: string
  summary: string
  files: CatalogFile[]
  preview_complete: boolean
  scan: {
    verdict: CatalogProvider['scan_verdict']
    policy: 'allow' | 'acknowledge' | 'block' | 'unavailable'
    summary: string
    findings: CatalogFinding[]
    complete: boolean
  }
  acknowledgements_required: string[]
  policy_revision: string | null
  content_digest: string | null
  setup_guidance: { id: string; summary: string; detail: string }[]
  confirmable: boolean
  review_token: string | null
  expires_at_ms: number | null
}

export interface CatalogOperation {
  operation_id: string
  identifier: string
  action: CatalogAction
  state: 'pending' | 'completed' | 'failed' | 'unknown'
  outcome: 'installed' | 'updated' | 'uninstalled' | 'unchanged' | 'refused' | null
  managed_state: ManagedState
  message: string
  installation_revision: string | null
  settlement?: 'settled' | 'reconciling'
}

export interface CatalogManagedInstallation {
  identifier: string
  target_relative_path: string
  installed_version: string | null
  latest_version: string | null
  managed_state: ManagedState
  installation_revision: string | null
  actions: CatalogAction[]
  settlement?: 'settled' | 'reconciling'
}

export interface CatalogPreview {
  entry: { identifier: string; provider: CatalogProvider }
  skill_md: string
  files: CatalogFile[]
  preview_complete: boolean
}
