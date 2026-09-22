import { lazy } from 'react'
import { useNavigate } from 'react-router'

import { openSession } from '@/app/open-session'
import { ROUTES_AREA, SIDEBAR_NAV_AREA, STARMAP_ROUTE } from '@/app/routes'
import { registry } from '@/contrib/registry'
import type { Contribution } from '@/contrib/types'
import { setSessionOwnerHint } from '@/store/session'

import type { InspectorSessionTarget } from './components/entity-inspector'
import type { LeaderOwner } from './leader-sessions'

const LazyLunarCity = lazy(async () => ({ default: (await import('./index')).LunarCity }))

export const LUNAR_CITY_ROUTE = '/lunar-city'

// The bundled Kanban plugin owns order 50. Lunar City is its immediate
// successor among contributed navigation rows while remaining independent of
// whether Kanban is enabled for a particular profile.
export const LUNAR_CITY_NAV_ORDER = 60

export function openLeaderFullChat(
  storedId: string,
  owner: LeaderOwner,
  navigate: ReturnType<typeof useNavigate>
): void {
  // Preserve the exact source/profile route while handing off to the normal
  // full chat. The ordinary session lifecycle resumes this durable id; no
  // Lunar City chat route or ambient gateway is introduced here.
  setSessionOwnerHint(storedId, owner)
  openSession(storedId, navigate, 'main')
}

export function openEntitySession(target: InspectorSessionTarget, navigate: ReturnType<typeof useNavigate>): void {
  const storedId = target.storedSessionId ?? target.sessionId
  const owner = Object.freeze({ connectionId: target.connectionId, profile: target.profile })

  setSessionOwnerHint(storedId, owner)
  openSession(storedId, navigate, 'main')
}

export function LunarCityRoute() {
  const navigate = useNavigate()

  return (
    <LazyLunarCity
      onOpenEntitySession={target => openEntitySession(target, navigate)}
      onOpenFullChat={(storedId, owner) => openLeaderFullChat(storedId, owner, navigate)}
      onOpenMemoryGraph={() => navigate(STARMAP_ROUTE)}
    />
  )
}

export const LUNAR_CITY_CONTRIBUTIONS: Contribution[] = [
  {
    id: 'lunar-city:page',
    area: ROUTES_AREA,
    data: { path: LUNAR_CITY_ROUTE },
    render: () => <LunarCityRoute />
  },
  {
    id: 'lunar-city:nav',
    area: SIDEBAR_NAV_AREA,
    order: LUNAR_CITY_NAV_ORDER,
    data: { codicon: 'globe', label: 'Lunar City', path: LUNAR_CITY_ROUTE }
  }
]

export function registerLunarCityContributions(): () => void {
  return registry.registerMany(LUNAR_CITY_CONTRIBUTIONS)
}
