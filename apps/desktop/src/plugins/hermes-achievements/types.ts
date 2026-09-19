export type AchievementState = 'discovered' | 'secret' | 'unlocked'

export interface AchievementTier {
  name: string
  threshold: number
}

export interface AchievementRequirement {
  gte: number
  metric: string
}

export interface AchievementEvidence {
  session_id?: string | null
  title?: string | null
  value?: number
}

export interface Achievement {
  category: string
  criteria: string
  description: string
  discovered: boolean
  evidence?: AchievementEvidence | null
  icon: string
  id: string
  kind: 'best_session' | 'lifetime' | 'multi_condition'
  name: string
  next_threshold: number
  next_tier: string | null
  progress: number
  progress_pct: number
  requirements?: AchievementRequirement[]
  secret?: boolean
  state: AchievementState
  threshold_metric?: string
  tier: string | null
  tiers?: AchievementTier[]
  unlocked: boolean
  unlocked_at?: number | null
}

export type ScanState = 'failed' | 'idle' | 'running' | string
export type ScanMode = 'failed' | 'full' | 'incremental' | 'in_progress' | 'pending' | string

export interface ScanStatus {
  finished_at?: number | null
  last_duration_ms?: number | null
  last_error?: string | null
  run_count?: number
  snapshot_age_seconds?: number | null
  snapshot_generated_at?: number | null
  snapshot_stale?: boolean
  started_at?: number | null
  state?: ScanState
  ttl_seconds?: number
}

export interface ScanMeta {
  mode?: ScanMode
  sessions_expected_total?: number | null
  sessions_rescanned?: number
  sessions_reused?: number
  sessions_scanned_so_far?: number | null
  sessions_total?: number
  status?: ScanStatus
}

export interface AchievementsResponse {
  achievements: Achievement[]
  discovered_count: number
  error?: string | null
  generated_at: number
  is_stale: boolean
  scan_meta: ScanMeta
  secret_count: number
  total_count: number
  unlocked_count: number
}

export interface RescanResponse extends AchievementsResponse {
  aggregate?: Record<string, unknown>
  ok: boolean
  sessions?: Array<Record<string, unknown>>
}
