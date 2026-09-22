import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import {
  $activeConnectionId,
  $connectionsRegistry,
  $hasMultipleConnections,
  selectConnection
} from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $profileColors,
  $profileCreateRequest,
  $profileOrder,
  $profiles,
  $showAllProfiles,
  ALL_PROFILES,
  normalizeProfileKey,
  profileLabel,
  refreshActiveProfile,
  selectProfile,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import { runImportProfileFlow } from '@/store/profile-share'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

import { ConnectionGlyph } from './connection-glyph'
import { buildRestGroups, type FleetAgent, fleetRouteKey } from './fleet-rail'
import { useFleetRoster } from './use-fleet-roster'
import { useProfilePrewarm } from './use-profile-prewarm'

/**
 * The profile picker that sits beside the gateway switcher in the statusbar
 * while the colored rail is hidden — the same choices the rail offers (this
 * gateway's profiles, every other gateway's agents in fleet mode, new / import
 * / manage) in a dropdown that reads like its neighbour. The rail keeps the
 * gestures that need squares (drag-order, hold-to-recolor); this is the
 * plain-dropdown door for people who run profiles as bots and don't want a
 * strip of them.
 */
export function ProfileSwitcher({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n()
  const p = t.profiles
  const navigate = useNavigate()
  const profiles = useStore($profiles)
  const order = useStore($profileOrder)
  const colors = useStore($profileColors)
  const gatewayProfile = useStore($activeGatewayProfile)
  const showAll = useStore($showAllProfiles)
  const multipleConnections = useStore($hasMultipleConnections)
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const roster = useStore($fleetRoster)
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRoute, setPendingRoute] = useState<null | string>(null)

  useFleetRoster(multipleConnections)

  // The `profile.create` hotkey bumps this request atom; the rail answers it
  // while mounted, so with the rail hidden this picker owns the dialog instead.
  const createRequest = useStore($profileCreateRequest)
  const lastCreateRef = useRef(createRequest)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (createRequest === lastCreateRef.current) {
      return
    }

    lastCreateRef.current = createRequest
    setCreateOpen(true)
  }, [createRequest])

  const connections = registry?.connections

  const restGroups = useMemo(
    () =>
      multipleConnections ? buildRestGroups({ activeConnectionId, connections: connections ?? [], order, roster }) : [],
    [activeConnectionId, connections, multipleConnections, order, roster]
  )

  const activeKey = normalizeProfileKey(gatewayProfile)
  const defaultProfile = profiles.find(profile => profile.is_default)

  const named = sortByProfileOrder(
    profiles.filter(profile => !profile.is_default),
    order
  )

  const ordered = defaultProfile ? [defaultProfile, ...named] : named
  const active = showAll ? undefined : profiles.find(profile => normalizeProfileKey(profile.name) === activeKey)
  const value = showAll ? ALL_PROFILES : (active?.name ?? '')

  const choose = (name: string) => {
    triggerHaptic('selection')

    if (name === ALL_PROFILES) {
      setShowAllProfiles(true)
    } else {
      selectProfile(name)
    }
  }

  const switchToRest = (agent: FleetAgent) => {
    const key = fleetRouteKey(agent.connectionId, agent.profile)
    triggerHaptic('selection')
    setPendingRoute(key)

    void selectConnection(agent.connectionId, { profile: agent.profile })
      .catch((error: unknown) => notifyError(error, p.switchConnectionFailed(agent.connectionLabel)))
      .finally(() => setPendingRoute(current => (current === key ? null : current)))
  }

  const triggerLabel = showAll ? p.allProfiles : active ? profileLabel(active) : p.title

  return (
    <div
      aria-busy={pendingRoute !== null}
      aria-label={p.title}
      className={cn('min-w-16 shrink overflow-hidden', compact ? 'h-full max-w-40' : 'max-w-72')}
      data-slot="profile-switcher"
      role="group"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${p.title}: ${triggerLabel}`}
            className={cn(
              'w-full min-w-0 justify-between overflow-hidden px-1 text-(--ui-text-secondary) data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground',
              compact && 'h-full min-h-0 rounded-none px-1.5 text-[0.6875rem] font-normal'
            )}
            size="xs"
            type="button"
            variant="ghost"
          >
            <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
              {pendingRoute !== null ? (
                <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />
              ) : showAll || !active ? (
                <Codicon aria-hidden="true" className="shrink-0" name="layers" size="0.75rem" />
              ) : (
                <ProfileGlyph
                  aria-hidden="true"
                  className="size-3 text-[0.4375rem]"
                  color={resolveProfileColor(active.name, colors)}
                  isDefault={active.is_default}
                  name={active.name}
                />
              )}
              <span className="truncate">{triggerLabel}</span>
            </span>
            <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-48 max-w-72" collisionPadding={8} side="top">
          <DropdownMenuRadioGroup onValueChange={choose} value={value}>
            {ordered.map(profile => (
              <ProfileItem
                color={resolveProfileColor(profile.name, colors)}
                isDefault={profile.is_default}
                key={profile.name}
                label={profileLabel(profile)}
                name={profile.name}
              />
            ))}
            {/* Nothing to widen to with one profile: the ALL view only exists
                once a second profile does (the rail hides its toggle then too). */}
            {profiles.length > 1 && (
              <DropdownMenuRadioItem className="min-w-0" value={ALL_PROFILES}>
                <span className="flex min-w-0 items-center gap-1.5">
                  <Codicon aria-hidden="true" className="text-(--ui-text-tertiary)" name="layers" size="0.875rem" />
                  <span className="truncate">{p.allProfiles}</span>
                </span>
              </DropdownMenuRadioItem>
            )}
          </DropdownMenuRadioGroup>
          {restGroups.map(group => (
            <div data-connection-id={group.connectionId} data-slot="profile-switcher-gateway" key={group.connectionId}>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className={cn(dropdownMenuSectionLabel, 'flex items-center gap-1.5')}>
                <ConnectionGlyph connection={group} />
                <span className="truncate">{group.label}</span>
                {!group.reachable && (
                  <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-amber-500" />
                )}
              </DropdownMenuLabel>
              {[group.defaultAgent, ...group.named].map(agent => (
                <DropdownMenuItem
                  aria-label={p.fleet.onGateway(agent.profile, group.label)}
                  className="min-w-0"
                  key={agent.profile}
                  onSelect={() => switchToRest(agent)}
                >
                  <span className="flex min-w-0 items-center gap-1.5">
                    <ProfileGlyph
                      aria-hidden="true"
                      color={resolveProfileColor(agent.profile, colors)}
                      isDefault={agent.isDefault}
                      name={agent.profile}
                    />
                    <span className="truncate">{agent.profile}</span>
                  </span>
                </DropdownMenuItem>
              ))}
            </div>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => setCreateOpen(true)}>
            <Codicon aria-hidden="true" name="add" size="0.875rem" />
            <span className="truncate">{p.newProfile}</span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => void runImportProfileFlow()}>
            <Codicon aria-hidden="true" name="cloud-download" size="0.875rem" />
            <span className="truncate">{p.importProfile}</span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => navigate(PROFILES_ROUTE)}>
            <Codicon aria-hidden="true" name="settings-gear" size="0.875rem" />
            <span className="truncate">{p.manageProfiles}</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <CreateProfileDialog
        onClose={() => setCreateOpen(false)}
        onCreated={async name => {
          await refreshActiveProfile()
          selectProfile(name)
        }}
        open={createOpen}
        profiles={profiles}
      />
    </div>
  )
}

function ProfileItem({
  color,
  isDefault,
  label,
  name
}: {
  color: null | string
  isDefault: boolean
  label: string
  name: string
}) {
  const { cancelPrewarm, notePointerMove, startPrewarm } = useProfilePrewarm(name)

  return (
    <DropdownMenuRadioItem
      className="min-w-0"
      onPointerEnter={startPrewarm}
      onPointerLeave={cancelPrewarm}
      onPointerMove={notePointerMove}
      value={name}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <ProfileGlyph aria-hidden="true" color={color} isDefault={isDefault} name={name} />
        <span className="truncate">{label}</span>
      </span>
    </DropdownMenuRadioItem>
  )
}
