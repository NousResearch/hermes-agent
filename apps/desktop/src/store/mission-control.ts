import type { WorkstreamFilterRuntime } from '@/store/workstream-filter'
import { workstreamActivityForSession } from '@/store/workstream-filter'
import type { WorkstreamLifecycle } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

export type MissionControlBucket = 'active' | 'blocked' | 'closed' | 'restart' | 'review' | 'safe_delete'

export interface MissionControlEntry {
  bucket: MissionControlBucket
  icon: string
  label: string
  lifecycle: WorkstreamLifecycle
  session: SessionInfo
  stateLabel: string
}

export type MissionControlBuckets = Record<MissionControlBucket, MissionControlEntry[]>

export const MISSION_CONTROL_BUCKETS = ['active', 'blocked', 'review', 'restart', 'closed', 'safe_delete'] as const

export const MISSION_CONTROL_BUCKET_META: Record<MissionControlBucket, { icon: string; label: string }> = {
  active: { icon: '✍️', label: 'Active' },
  blocked: { icon: '❗️', label: 'Blocked' },
  closed: { icon: '✅', label: 'Closed' },
  restart: { icon: '⚡️', label: 'Restart' },
  review: { icon: '🔎', label: 'Review' },
  safe_delete: { icon: '📁', label: 'Safe delete' }
}

const BLOCKED_STATES = new Set(['blocked', 'warn'])
const REVIEW_STATES = new Set(['plan_review', 'verify'])

export function emptyMissionControlBuckets(): MissionControlBuckets {
  return {
    active: [],
    blocked: [],
    closed: [],
    restart: [],
    review: [],
    safe_delete: []
  }
}

export function missionControlBucketFor(lifecycle: WorkstreamLifecycle, state: string): MissionControlBucket {
  if (lifecycle === 'safe_delete') {
    return 'safe_delete'
  }

  if (lifecycle === 'closed') {
    return 'closed'
  }

  if (lifecycle === 'restart_required' || state === 'restart') {
    return 'restart'
  }

  if (BLOCKED_STATES.has(state)) {
    return 'blocked'
  }

  if (REVIEW_STATES.has(state)) {
    return 'review'
  }

  return 'active'
}

export function buildMissionControlBuckets(
  sessions: readonly SessionInfo[],
  runtime: WorkstreamFilterRuntime
): MissionControlBuckets {
  const buckets = emptyMissionControlBuckets()

  for (const session of sessions) {
    const { activity, lifecycle } = workstreamActivityForSession(session, runtime)
    const bucket = missionControlBucketFor(lifecycle, activity.state)

    buckets[bucket].push({
      bucket,
      icon: activity.icon,
      label: activity.label,
      lifecycle,
      session,
      stateLabel: activity.label
    })
  }

  return buckets
}
