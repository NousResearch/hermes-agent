import { useCallback, useEffect, useRef, useState } from 'react'

import type { LunarCitySnapshot, LunarCityWorldHandle, LunarEntity, QualityTier } from '../model'
import { $lunarCitySnapshot } from '../store'

import { entityFriendlyLabel } from './entity-list'

export type LunarCityBadgeStatus = 'blocked' | 'delivering' | 'resting' | 'review' | 'waiting' | 'working'

export interface LunarCityStatusBadge {
  entity: LunarEntity
  label: string
  status: LunarCityBadgeStatus
}

export interface ProjectedLunarCityStatusBadge extends LunarCityStatusBadge {
  x: number
  y: number
}

const BADGE_STATES: Readonly<Record<string, LunarCityBadgeStatus>> = {
  blocked: 'blocked',
  error: 'blocked',
  failed: 'blocked',
  triage: 'blocked',
  review: 'review',
  under_review: 'review',
  awaiting_approval: 'review',
  delivering: 'delivering',
  handoff: 'delivering',
  dependency: 'delivering',
  orchestration: 'delivering',
  rest: 'resting',
  resting: 'resting',
  sleep: 'resting',
  sleeping: 'resting',
  heartbeat: 'resting',
  pause: 'resting',
  paused: 'resting',
  recovery: 'resting',
  wait: 'waiting',
  waiting: 'waiting',
  queued: 'waiting',
  queue: 'waiting',
  ready: 'waiting',
  pending: 'waiting',
  resource_wait: 'waiting',
  waiting_for_resource: 'waiting',
  work: 'working',
  working: 'working',
  running: 'working'
}

export function badgeStatusForEntity(entity: LunarEntity): LunarCityBadgeStatus | undefined {
  if (entity.authority !== 'authoritative') {
    return undefined
  }

  const source = (entity.sourceState ?? entity.animation)
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/gu, '_')

  return BADGE_STATES[source]
}

export function statusBadgeCandidates(snapshot: LunarCitySnapshot, limit = 8): readonly LunarCityStatusBadge[] {
  return [...snapshot.entities.values()]
    .map(entity => {
      const status = badgeStatusForEntity(entity)

      return status ? { entity, label: entityFriendlyLabel(entity), status } : undefined
    })
    .filter((badge): badge is LunarCityStatusBadge => badge !== undefined && badge.entity.position !== undefined)
    .sort((left, right) => {
      const priority = { blocked: 0, review: 1, delivering: 2, working: 3, waiting: 4, resting: 5 }

      return priority[left.status] - priority[right.status] || left.entity.key.localeCompare(right.entity.key)
    })
    .slice(0, Math.max(0, limit))
}

interface LunarCityStatusBadgesProps {
  onSelect(entity: LunarEntity): void
  qualityTier: QualityTier
  rendererStatus: 'loading' | 'ready' | 'degraded' | 'unavailable'
  worldRef: { current: LunarCityWorldHandle | undefined }
}

/**
 * A deliberately cheap Phase 3 surface: at most eight DOM pins, sampled at
 * 10Hz, and only while the selected quality tier keeps decorations enabled.
 * The canvas remains the source of visual truth; these pins are an accessible
 * operational affordance, not a second renderer.
 */
export function LunarCityStatusBadges({ onSelect, qualityTier, rendererStatus, worldRef }: LunarCityStatusBadgesProps) {
  const [badges, setBadges] = useState<readonly ProjectedLunarCityStatusBadge[]>([])

  const published = useRef<readonly ProjectedLunarCityStatusBadge[]>([])

  const publish = useCallback((next: readonly ProjectedLunarCityStatusBadge[]) => {
    const current = published.current

    const unchanged = current.length === next.length && current.every((badge, index) => {
      const other = next[index]!

      return badge.entity.key === other.entity.key && badge.label === other.label && badge.status === other.status &&
        badge.x === other.x && badge.y === other.y
    })

    if (!unchanged) { published.current = next; setBadges(next) }
  }, [])

  const refresh = useCallback(() => {
    if (document.hidden || qualityTier === 'efficient' || rendererStatus !== 'ready') {
      publish([])

      return
    }

    const world = worldRef.current
    const project = world?.projectWorldPoint

    if (!project) { publish([]);

 return }

    const projected = statusBadgeCandidates($lunarCitySnapshot.get()).flatMap(candidate => {
      const point = candidate.entity.position
      const screen = point ? project.call(world, point) : undefined

      return screen?.visible ? [{ ...candidate, x: screen.x, y: screen.y }] : []
    })

    publish(projected)
  }, [qualityTier, rendererStatus, worldRef, publish])

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined

    const visibility = () => {
      if (timer !== undefined) {clearInterval(timer)}
      timer = undefined
      refresh()

      if (!document.hidden && qualityTier !== 'efficient' && rendererStatus === 'ready') {timer = setInterval(refresh, 100)}
    }

    visibility()
    document.addEventListener('visibilitychange', visibility)

    return () => {
      if (timer !== undefined) {clearInterval(timer)}
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [qualityTier, refresh, rendererStatus])

  if (badges.length === 0) {
    return null
  }

  return (
    <div
      aria-label="Visible worker status pins"
      className="lunar-city-status-badges"
      data-testid="lunar-city-status-badges"
    >
      {badges.map(badge => (
        <button
          aria-label={`${badge.label}: ${badge.status}`}
          className={`lunar-city-status-badge lunar-city-status-badge-${badge.status}`}
          key={badge.entity.key}
          onClick={() => {
            const current = $lunarCitySnapshot.get().entities.get(badge.entity.key)

            if (current && badgeStatusForEntity(current)) {onSelect(current)}
          }}
          style={{ left: `${badge.x}%`, top: `${badge.y}%` }}
          type="button"
        >
          <span aria-hidden="true" className="lunar-city-status-badge-pin" />
          <span>{badge.status}</span>
        </button>
      ))}
    </div>
  )
}
