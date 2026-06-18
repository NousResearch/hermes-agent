import type { RouteAdvisoryResponse } from '@/types/hermes'

const DEFAULT_ROUTE_IDS = new Set(['main-hermes', 'macos', 'default'])

function routeName(advisory: RouteAdvisoryResponse): string {
  return String(advisory.route_id || advisory.profile || '').trim()
}

export function shouldDisplayRouteAdvisory(advisory: null | RouteAdvisoryResponse | undefined): advisory is RouteAdvisoryResponse {
  if (!advisory || advisory.error) {
    return false
  }

  if (advisory.advisory_mode !== true || advisory.auto_execute !== false) {
    return false
  }

  const route = routeName(advisory).toLowerCase()
  const profile = String(advisory.profile || '').trim().toLowerCase()

  if (!route || DEFAULT_ROUTE_IDS.has(route) || DEFAULT_ROUTE_IDS.has(profile)) {
    return false
  }

  return Number(advisory.confidence ?? 0) > 0
}

export function RouteAdvisoryBanner({ advisory }: { advisory: null | RouteAdvisoryResponse | undefined }) {
  if (!shouldDisplayRouteAdvisory(advisory)) {
    return null
  }

  const profile = advisory.profile || advisory.route_id || 'specialist profile'
  const confidence = Number(advisory.confidence ?? 0)
  const blockedActions = advisory.blocked_actions?.filter(Boolean) ?? []

  return (
    <div
      aria-label="Route advisory"
      className="rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_34%,transparent)] bg-accent/14 px-3 py-2 text-[0.72rem] leading-snug text-foreground shadow-sm"
      role="status"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-medium">Recommended profile: {profile}</span>
        <span className="text-muted-foreground">confidence {Number.isInteger(confidence) ? confidence : confidence.toFixed(1)}</span>
        <span className="rounded-full border border-[color-mix(in_srgb,var(--dt-composer-ring)_26%,transparent)] px-2 py-0.5 text-[0.66rem] uppercase tracking-wide text-muted-foreground">
          Advisory only
        </span>
      </div>
      <div className="mt-1 text-muted-foreground">No profile switch or auto-execution will happen from this banner.</div>
      {blockedActions.length > 0 && (
        <div className="mt-1 text-muted-foreground">Blocked actions: {blockedActions.join(', ')}</div>
      )}
    </div>
  )
}
