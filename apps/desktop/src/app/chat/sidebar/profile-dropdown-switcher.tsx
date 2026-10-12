import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
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
  $newChatProfile,
  $newChatRoute,
  $profileColors,
  $profileCreateRequest,
  $profileOrder,
  $profiles,
  $showAllProfiles,
  ALL_PROFILES,
  changeDraftAgent,
  changeDraftProfile,
  normalizeProfileKey,
  prewarmProfilePick,
  profileLabel,
  refreshActiveProfile,
  selectProfile,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import { $profileDotStateByScope, profileDotSummaryFor } from '@/store/profile-dot-state'
import { runImportProfileFlow } from '@/store/profile-share'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

import { FleetGatewayMenuGroup } from './fleet-gateway-menu-group'
import { buildRestGroups, type FleetAgent, fleetRouteKey } from './fleet-rail'
import { useLocalDeviceSwitch } from './local-device-switch'
import { ProfileLaunchContextMenu } from './profile-launch-menu'
import { combineProfileStatus, ProfileStatusDot, profileStatusLabel, useProfileStatus } from './profile-status'
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
interface ProfileSwitcherProps {
  compact?: boolean
  placement?: 'composer' | 'sidebar' | 'statusbar'
}

export function ProfileSwitcher({ compact = false, placement = 'statusbar' }: ProfileSwitcherProps) {
  const composer = placement === 'composer'
  const look = placementLook(placement, compact)
  const { t } = useI18n()
  const p = t.profiles
  const navigate = useNavigate()
  const profiles = useStore($profiles)
  const order = useStore($profileOrder)
  const colors = useStore($profileColors)
  const gatewayProfile = useStore($activeGatewayProfile)
  const draftProfile = useDraftProfile()
  const allProfiles = useStore($showAllProfiles)
  const showAll = !composer && allProfiles
  const multipleConnections = useStore($hasMultipleConnections)
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const roster = useStore($fleetRoster)
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRoute, setPendingRoute] = useState<null | string>(null)
  const { dialog: localDeviceDialog, request: requestLocalDevice } = useLocalDeviceSwitch()

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

  const active = showAll
    ? undefined
    : profiles.find(profile => normalizeProfileKey(profile.name) === (composer ? draftProfile || activeKey : activeKey))

  const value = showAll ? ALL_PROFILES : (active?.name ?? '')

  const choose = (name: string) => {
    triggerHaptic('selection')

    if (composer) {
      changeDraftProfile(name)
    } else if (name === ALL_PROFILES) {
      setShowAllProfiles(true)
    } else {
      selectProfile(name)
    }
  }

  const commitRestSwitch = (agent: FleetAgent) => {
    if (composer) {
      changeDraftAgent({ connectionId: agent.connectionId, profile: agent.profile })

      return
    }

    const key = fleetRouteKey(agent.connectionId, agent.profile)
    triggerHaptic('selection')
    setPendingRoute(key)

    void selectConnection(agent.connectionId, { profile: agent.profile })
      .catch((error: unknown) => notifyError(error, p.switchConnectionFailed(agent.connectionLabel)))
      .finally(() => setPendingRoute(current => (current === key ? null : current)))
  }

  const switchToRest = (agent: FleetAgent) => {
    if (agent.connectionKind !== 'local') {
      commitRestSwitch(agent)

      return
    }

    const key = fleetRouteKey(agent.connectionId, agent.profile)
    setPendingRoute(key)

    void requestLocalDevice({
      connectionId: agent.connectionId,
      label: agent.connectionLabel,
      profile: agent.profile,
      replaceCenter: agent.profile === 'default'
    }).then(accepted => {
      setPendingRoute(current => (current === key ? null : current))

      if (accepted) {
        commitRestSwitch(agent)
      }
    })
  }

  const statuses = useStore($profileDotStateByScope)

  const otherProfileStatus = combineProfileStatus([
    ...ordered
      .filter(profile => showAll || normalizeProfileKey(profile.name) !== activeKey)
      .map(profile => profileDotSummaryFor(statuses, activeConnectionId, profile.name)),
    ...restGroups.flatMap(group =>
      [group.defaultAgent, ...group.named].map(agent =>
        profileDotSummaryFor(statuses, agent.connectionId, agent.profile)
      )
    )
  ])

  const triggerLabel = showAll
    ? p.allProfiles
    : active
      ? profileLabel(active)
      : composer
        ? draftProfile || gatewayProfile
        : p.title

  return (
    <div
      aria-busy={pendingRoute !== null}
      aria-label={p.title}
      className={cn('min-w-16 shrink overflow-hidden', look.widthClass)}
      data-slot="profile-switcher"
      role="group"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={withProfileStatus(`${p.title}: ${triggerLabel}`, p, otherProfileStatus)}
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
                  defaultStyle={look.glyphStyle}
                  isDefault={active.is_default}
                  name={look.glyphName(active)}
                />
              )}
              <span className="truncate">{triggerLabel}</span>
            </span>
            {otherProfileStatus && <ProfileStatusDot summary={otherProfileStatus} />}
            <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-48 max-w-72" collisionPadding={8} side={look.menuSide}>
          <DropdownMenuRadioGroup onValueChange={choose} value={value}>
            {ordered.map(profile => (
              <ProfileItem
                color={resolveProfileColor(profile.name, colors)}
                connectionId={activeConnectionId}
                defaultStyle={look.glyphStyle}
                isDefault={profile.is_default}
                key={profile.name}
                label={profileLabel(profile)}
                name={profile.name}
              />
            ))}
            {/* Nothing to widen to with one profile: the ALL view only exists
                once a second profile does (the rail hides its toggle then too). */}
            {!composer && profiles.length > 1 && (
              <DropdownMenuRadioItem className="min-w-0" value={ALL_PROFILES}>
                <span className="flex min-w-0 items-center gap-1.5">
                  <Codicon aria-hidden="true" className="text-(--ui-text-tertiary)" name="layers" size="0.875rem" />
                  <span className="truncate">{p.allProfiles}</span>
                </span>
              </DropdownMenuRadioItem>
            )}
          </DropdownMenuRadioGroup>
          {restGroups.map(group => (
            <FleetGatewayMenuGroup
              group={group}
              key={group.connectionId}
              onSelect={switchToRest}
              slot="profile-switcher-gateway"
              wrapRow={(row, agent, label) => (
                <ProfileLaunchContextMenu
                  connectionId={agent.connectionId}
                  key={fleetRouteKey(agent.connectionId, agent.profile)}
                  label={label}
                  profile={agent.profile}
                >
                  {row}
                </ProfileLaunchContextMenu>
              )}
            />
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
      {localDeviceDialog}
    </div>
  )
}

/** How the picker presents itself where it is mounted: the statusbar keeps its
 *  compact home-glyph form; the Simple sidebar and composer name profiles by
 *  label and initial. */
function placementLook(placement: NonNullable<ProfileSwitcherProps['placement']>, compact: boolean) {
  const labeled = placement !== 'statusbar'
  const sidebar = placement === 'sidebar'

  return {
    glyphName: (profile: Parameters<typeof profileLabel>[0]) => (labeled ? profileLabel(profile) : profile.name),
    glyphStyle: labeled ? ('initial' as const) : ('home' as const),
    menuSide: sidebar ? ('bottom' as const) : ('top' as const),
    widthClass: sidebar ? 'w-full' : compact ? 'h-full max-w-40' : 'max-w-72'
  }
}

/** The owner a new chat is about to use: its captured route, else the intended profile. */
function useDraftProfile() {
  const route = useStore($newChatRoute)
  const intent = useStore($newChatProfile)

  return route?.targetProfile || route?.profile || intent
}

const withProfileStatus = (
  label: string,
  copy: Parameters<typeof profileStatusLabel>[0],
  summary: Parameters<typeof profileStatusLabel>[1] | null | undefined
) => (summary ? `${label}, ${profileStatusLabel(copy, summary)}` : label)

function ProfileItem({
  connectionId,
  color,
  defaultStyle,
  isDefault,
  label,
  name
}: {
  connectionId: null | string
  color: null | string
  defaultStyle: 'home' | 'initial'
  isDefault: boolean
  label: string
  name: string
}) {
  const { t } = useI18n()
  const summary = useProfileStatus(name, connectionId)
  const { cancelPrewarm, notePointerMove, startPrewarm } = useProfilePrewarm(name, prewarmProfilePick)

  return (
    <ProfileLaunchContextMenu connectionId={connectionId ?? 'local'} label={label} profile={name}>
      <DropdownMenuRadioItem
        aria-label={withProfileStatus(label, t.profiles, summary)}
        className="min-w-0"
        onPointerEnter={startPrewarm}
        onPointerLeave={cancelPrewarm}
        onPointerMove={notePointerMove}
        value={name}
      >
        <span className="flex min-w-0 items-center gap-1.5">
          <ProfileGlyph
            aria-hidden="true"
            color={color}
            defaultStyle={defaultStyle}
            isDefault={isDefault}
            name={defaultStyle === 'initial' ? label : name}
          />
          <span className="truncate">{label}</span>
          {summary && <ProfileStatusDot summary={summary} />}
        </span>
      </DropdownMenuRadioItem>
    </ProfileLaunchContextMenu>
  )
}
