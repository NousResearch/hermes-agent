import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { ConnectionGlyph } from '@/app/chat/sidebar/connection-glyph'
import { buildRestGroups } from '@/app/chat/sidebar/fleet-rail'
import { useFleetRoster } from '@/app/chat/sidebar/use-fleet-roster'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuSectionLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile, $profileColors, $profiles, refreshProfiles } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsRequestProfile, $settingsScopeProfile, setSettingsScope } from '@/store/settings-scope'

/** A settings-only owner pick: no profile activation, prewarming or sidebar actions. */
export function SettingsProfileScope({ className }: { className?: string }) {
  const { t } = useI18n()
  const scope = t.settings.profileScope
  const selected = useStore($settingsRequestProfile)
  const profile = useStore($settingsScopeProfile)
  const profiles = useStore($profiles)
  const activeProfile = useStore($activeGatewayProfile)
  const colors = useStore($profileColors)
  const registry = useStore($connectionsRegistry)
  const roster = useStore($fleetRoster)
  const connection = useStore($connection)
  const registered = Boolean(registry?.connections.length)
  useFleetRoster(registered)

  useEffect(() => {
    void refreshProfiles().catch(() => undefined)
  }, [])

  // A resolved local primary belongs to This device even without registryScoped.
  // This identity is display-only: ambient requests must retain profile overrides.
  const activeId = connection?.registryScoped
    ? connection.connectionId
    : connection?.mode === 'local'
      ? registry?.connections.find(item => item.kind === 'local')?.id
      : null

  const connectionId = selected && typeof selected === 'object' ? selected.connectionId : (activeId ?? null)
  const selectedGateway = registry?.connections.find(item => item.id === connectionId)
  const label = connectionId ? t.profiles.fleet.onGateway(profile, selectedGateway?.label ?? connectionId) : profile
  const value = JSON.stringify([connectionId, profile])

  const groups = buildRestGroups({
    activeConnectionId: null,
    connections: registry?.connections ?? [],
    roster: roster?.sources.length ? roster : null
  })

  // The active roster can be fresher than fleet enumeration. Only merge it into
  // its proven registered owner; an unregistered primary must never become local.
  const legacy = !activeId
  const legacyProfiles = Array.from(new Set([activeProfile || 'default', ...profiles.map(item => item.name)]))

  if (activeId && !groups.some(group => group.connectionId === activeId)) {
    groups.unshift(
      ...buildRestGroups({
        activeConnectionId: null,
        connections: registry?.connections.filter(item => item.id === activeId) ?? [],
        roster: null
      })
    )
  }

  const row = (name: string, owner: string | null, gatewayLabel?: string) => (
    <DropdownMenuRadioItem
      aria-label={gatewayLabel ? t.profiles.fleet.onGateway(name, gatewayLabel) : name}
      key={JSON.stringify([owner, name])}
      onSelect={() => setSettingsScope(owner ? { connectionId: owner, profile: name } : name)}
      value={JSON.stringify([owner, name])}
    >
      <ProfileGlyph
        aria-hidden="true"
        color={resolveProfileColor(name, colors)}
        isDefault={name === 'default'}
        name={name}
      />
      <span className="truncate">{name}</span>
    </DropdownMenuRadioItem>
  )

  return (
    <div className={cn('flex min-w-0 items-center gap-2', className)} data-slot="settings-profile-scope">
      <span className="shrink-0 text-xs text-(--ui-text-secondary)">{scope.appliesTo}</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${scope.appliesTo}: ${label}`}
            className="min-w-0 max-w-80 shrink"
            size="sm"
            variant="secondary"
          >
            <ProfileGlyph
              aria-hidden="true"
              color={resolveProfileColor(profile, colors)}
              isDefault={profile === 'default'}
              name={profile}
            />
            <span className="truncate">{label}</span>
            <Codicon aria-hidden="true" name="chevron-down" size="0.875rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className="max-h-[min(20rem,var(--radix-dropdown-menu-content-available-height))] min-w-56 max-w-80 overflow-y-auto"
          collisionPadding={8}
        >
          <DropdownMenuRadioGroup value={value}>
            {legacy && legacyProfiles.map(name => row(name, null))}
            {groups.map(group => {
              const names = Array.from(
                new Set([
                  group.defaultAgent.profile,
                  ...group.named.map(agent => agent.profile),
                  ...(group.connectionId === activeId ? profiles.map(item => item.name) : [])
                ])
              )

              return (
                <div data-connection-id={group.connectionId} key={group.connectionId}>
                  <DropdownMenuLabel className={cn(dropdownMenuSectionLabel, 'flex items-center gap-1.5')}>
                    <ConnectionGlyph connection={group} />
                    <span className="truncate">
                      {group.reachable ? group.label : t.profiles.fleet.gatewayUnreachable(group.label)}
                    </span>
                  </DropdownMenuLabel>
                  {names.map(name => row(name, group.connectionId, group.label))}
                </div>
              )
            })}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
