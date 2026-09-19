import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { DesktopRosterAgent } from '@/global'
import { getProfiles, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { activeGatewayConnectionId } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

interface ScopeOption {
  key: string
  label: string
  value: string
}

export interface CapabilityScope {
  /** Which profile's Skills/Tools/MCP config every tab reads and writes — and
   *  on WHICH gateway. */
  profile: ProfileScope
  /** Cache-key form of `profile`: every scoped query key ends with it, and the
   *  tabs are keyed on it so a scope change is a fresh tab. */
  key: string
  /** Scoped to a DIFFERENT backend than the window's active gateway? The MCP
   *  tab's live-reload RPC rides the active gateway socket, which would reload
   *  the wrong machine — it is withheld for cross-backend scopes. */
  crossBackend: boolean
  /** Display name of the selected profile (the Plugins tab's Agent column). */
  label?: string
  options: ScopeOption[]
  value: string
  onChange: (value: string) => void
}

/**
 * The Capabilities scope: which profile's config the page is editing, and on
 * which gateway. A profile belongs to one gateway, so on a multi-connection
 * desktop the selector offers every (connection, profile) pair from the union
 * agent roster and an explicit pick routes every read/write to that machine's
 * backend. Defaults to the app-wide active profile. A `fixedProfile` (+
 * optional `fixedConnection`) pins the scope outright, selector hidden.
 */
export function useCapabilityScope({
  fixedConnection,
  fixedProfile
}: {
  fixedConnection?: string
  fixedProfile?: string
}): CapabilityScope {
  const activeProfile = useStore($activeGatewayProfile)
  const [scopeOverride, setScopeOverride] = useState<null | string | { connectionId: string; profile: string }>(null)

  const scopeProfile: ProfileScope = useMemo(
    () =>
      fixedProfile
        ? fixedConnection
          ? { connectionId: fixedConnection, profile: fixedProfile }
          : fixedProfile
        : (scopeOverride ?? activeProfile ?? null),
    [activeProfile, fixedConnection, fixedProfile, scopeOverride]
  )

  const scopeKey = profileScopeKey(scopeProfile)

  // The registry connection an explicit override pins — null for the ambient
  // active-profile path and for fixed-profile pins without a connection.
  const scopeConnectionId =
    scopeProfile && typeof scopeProfile === 'object' ? (scopeProfile.connectionId ?? '').trim() || 'local' : null

  const { data: profilesData } = useQuery({
    queryKey: ['capabilities-profiles'],
    queryFn: () => getProfiles(),
    staleTime: 60_000,
    // Pinned scope never shows the selector, so don't fetch the roster for it.
    enabled: !fixedProfile
  })

  // v2 multi-connection registry: cheap local IPC to learn whether more than
  // one gateway is registered. Only then is the (heavier) union agent roster
  // fetched to feed the selector — single-connection setups keep the exact
  // legacy profiles list. Both feature-detected for older Electron mains.
  const registryBridge = window.hermesDesktop?.connections
  const rosterBridge = window.hermesDesktop?.getAgentRoster

  const { data: registryData } = useQuery({
    queryKey: ['capabilities-connections-registry'],
    queryFn: () => registryBridge!.list(),
    staleTime: 60_000,
    enabled: !fixedProfile && Boolean(registryBridge) && Boolean(rosterBridge)
  })

  const multiConnection = (registryData?.connections.length ?? 0) > 1

  const { data: rosterData } = useQuery({
    queryKey: ['capabilities-agent-roster'],
    queryFn: () => rosterBridge!(),
    staleTime: 60_000,
    enabled: !fixedProfile && multiConnection
  })

  // An app-wide profile switch drops any scope override — the user just
  // changed what "here" means, and a stale override pointing at the previous
  // selection would be surprising.
  useOnProfileSwitch(() => setScopeOverride(null))

  // Scope-selector rows. Multi-connection desktops list every reachable
  // (connection, profile) agent from the union roster — the selected profile
  // is configured ON ITS OWN GATEWAY. Otherwise the legacy per-profile list.
  const options: ScopeOption[] = useMemo(() => {
    if (multiConnection && rosterData?.agents?.length) {
      const activeId = activeGatewayConnectionId() ?? 'local'

      return rosterData.agents.map((agent: DesktopRosterAgent) => ({
        key: `${agent.connectionId}::${agent.profile}`,
        label:
          agent.connectionId === activeId
            ? `${agent.profile} — ${agent.connectionLabel} (current)`
            : `${agent.profile} — ${agent.connectionLabel}`,
        value: `${agent.connectionId}::${agent.profile}`
      }))
    }

    return (profilesData?.profiles ?? []).map(p => ({
      key: p.name,
      label: p.is_default ? 'Hermes (default)' : p.name,
      value: p.name
    }))
  }, [multiConnection, profilesData, rosterData])

  // The selector's current value must match one option's value exactly. On the
  // roster path an ambient (non-override) scope is the active gateway's
  // profile, which lives at `activeConnectionId::profile`.
  const value = useMemo(() => {
    if (scopeProfile && typeof scopeProfile === 'object') {
      return `${(scopeProfile.connectionId ?? '').trim() || 'local'}::${scopeProfile.profile ?? ''}`
    }

    if (multiConnection && rosterData?.agents?.length) {
      return `${activeGatewayConnectionId() ?? 'local'}::${normalizeProfileKey(scopeProfile)}`
    }

    return scopeProfile ?? ''
  }, [multiConnection, rosterData, scopeProfile])

  // Selector option values are `connectionId::profile` on multi-connection
  // desktops (the roster path) and bare profile names otherwise, so one
  // handler decodes both.
  const onChange = (picked: string) => {
    const sep = picked.indexOf('::')

    // Roster picks (`connectionId::profile`) stay objects — a `local::` pick
    // must PIN the local pool even while a remote gateway is active. Legacy
    // bare-name picks stay strings so cache keys and routing are unchanged.
    const next: ProfileScope =
      sep >= 0 ? { connectionId: picked.slice(0, sep), profile: picked.slice(sep + 2) } : picked

    if (profileScopeKey(next) === scopeKey) {
      return
    }

    setScopeOverride(next as string | { connectionId: string; profile: string })
  }

  return {
    crossBackend: scopeConnectionId !== null && scopeConnectionId !== (activeGatewayConnectionId() ?? 'local'),
    key: scopeKey,
    label: options.find(option => option.value === value)?.label,
    onChange,
    options,
    profile: scopeProfile,
    value
  }
}

/**
 * Scope selector, shown above EVERY Capabilities tab (Skills, Tools, MCP,
 * Plugins). Lets the user configure ANY profile's capabilities — on any
 * registered gateway — without switching the whole app. Only meaningful with
 * >1 option; hidden otherwise to avoid clutter.
 */
export function CapabilityScopeSelector({
  compact = false,
  scope
}: {
  /** In Plugins, the selector belongs only to the Agent column. */
  compact?: boolean
  scope: CapabilityScope
}) {
  const { t } = useI18n()

  if (scope.options.length <= 1) {
    return null
  }

  return (
    <div className={cn('flex min-w-0 items-center gap-2', compact ? 'flex-1' : 'border-b border-(--ui-stroke-secondary) px-3 py-2')}>
      {!compact && <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{t.skills.configuringProfile}</span>}
      <Select onValueChange={scope.onChange} value={scope.value}>
        <SelectTrigger className={cn('text-xs', compact ? 'h-6 w-full max-w-64 px-2' : 'h-7 w-56')}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {scope.options.map(option => (
            <SelectItem key={option.key} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
