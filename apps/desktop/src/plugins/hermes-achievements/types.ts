/**
 * Types for the hermes-achievements desktop plugin — mirrors the payloads the
 * dashboard plugin API returns (plugins/hermes-achievements/dashboard/plugin_api.py).
 */

export type AchievementState = 'unlocked' | 'discovered' | 'secret'

export interface AchievementEvidence {
  session_id: string
  title: string
  value: number
}

export interface Achievement {
  id: string
  name: string
  description: string
  category: string
  icon: string
  kind: string
  state: AchievementState
  unlocked: boolean
  discovered: boolean
  tier: string | null
  next_tier: string | null
  next_threshold: number
  progress: number
  progress_pct: number
  criteria?: string
  unlocked_at?: number
  evidence?: AchievementEvidence | null
}

export interface AchievementsResponse {
  achievements: Achievement[]
  unlocked_count: number
  discovered_count: number
  secret_count: number
  total_count: number
  error?: string | null
  generated_at: number
  is_stale: boolean
  scan_meta?: Record<string, unknown>
}

export type AchievementFilter = 'all' | 'unlocked' | 'discovered' | 'secret'

export const TIER_ORDER = ['Copper', 'Silver', 'Gold', 'Diamond', 'Olympian'] as const
