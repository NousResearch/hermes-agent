import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $activeConnectionId, $connectionsRegistry } from '@/store/connections'
import {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  $profileColors,
  $profileOrder,
  $profiles,
  $profilesByConnection,
  normalizeProfileKey,
  pinNewChatProfile,
  profileLabel,
  resolveNewChatOwnerRoute,
  sortByProfileOrder
} from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { defaultNewChatProfile, type ProfileSelectionMode } from './profile-selection'

const PILL = cn(
  'h-(--composer-control-size) min-w-0 shrink gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export interface ProfilePillOwner {
  connectionId?: null | string
  profile: null | string | undefined
}

export function ProfilePill({ mode, owner }: { mode: ProfileSelectionMode; owner: ProfilePillOwner }) {
  const { t } = useI18n()
  const activeProfile = useStore($activeGatewayProfile)
  const activeConnectionId = useStore($activeConnectionId)
  const profiles = useStore($profiles)
  const profilesByConnection = useStore($profilesByConnection)
  const colors = useStore($profileColors)
  const order = useStore($profileOrder)
  const registry = useStore($connectionsRegistry)
  const newChatProfile = useStore($newChatProfile)
  const newChatRoute = useStore($newChatRoute)
  const newChatConnectionId = useStore($newChatConnectionId)

  // Keep each registry source's profile labels separate. The same canonical
  // profile name can exist on two gateways with different display names.
  const draftProfile = newChatProfile ?? newChatRoute?.profile ?? defaultNewChatProfile()
  const newOwnerRoute = mode === 'started' ? null : resolveNewChatOwnerRoute(draftProfile)

  const profileKey = normalizeProfileKey(
    mode === 'started'
      ? owner.profile
      : newOwnerRoute?.targetProfile || newOwnerRoute?.profile || draftProfile || activeProfile
  )

  const connectionId =
    mode === 'started'
      ? (owner.connectionId ?? undefined)
      : (newOwnerRoute?.connectionId ?? newChatConnectionId ?? undefined)

  const ownerProfiles = connectionId
    ? (profilesByConnection.get(connectionId) ?? (activeConnectionId === connectionId ? profiles : []))
    : profiles

  const profile = ownerProfiles.find(item => normalizeProfileKey(item.name) === profileKey)
  const label = profile ? profileLabel(profile) : profileKey
  const visibleProfileName = label === profileKey ? profileKey : `${label} · ${profileKey}`

  const connectionLabel = connectionId
    ? (registry?.connections.find(connection => connection.id === connectionId)?.label ?? null)
    : null

  const menuConnectionLabel = activeConnectionId
    ? (registry?.connections.find(connection => connection.id === activeConnectionId)?.label ?? null)
    : null

  const visibleOwner = connectionLabel ? `${visibleProfileName} · ${connectionLabel}` : visibleProfileName

  const accessibleLabel =
    mode === 'started' ? t.sidebar.row.ownedByProfile(visibleOwner) : `${t.profiles.title}: ${visibleOwner}`

  const orderedProfiles = sortProfiles(profiles, order)
  const canChoose = mode === 'draft' && orderedProfiles.length > 1
  const menuValue = !newOwnerRoute?.connectionId || newOwnerRoute.connectionId === activeConnectionId ? profileKey : ''

  const selectProfile = (name: string) => {
    const target = orderedProfiles.find(item => item.name === name)

    if (target) {
      // This pins only the next new-chat route. It neither activates a gateway
      // nor clears/resumes an existing session.
      pinNewChatProfile(target.name)
    }
  }

  const glyph = (
    <ProfileGlyph
      aria-hidden="true"
      className="size-3.5 text-[0.4375rem]"
      color={resolveProfileColor(profileKey, colors)}
      isDefault={profileKey === 'default'}
      name={profileKey}
    />
  )

  if (!canChoose) {
    return (
      <Tip label={visibleOwner} placement="control">
        <span
          aria-label={mode === 'started' ? accessibleLabel : `${t.profiles.title}: ${visibleOwner}`}
          className="inline-flex h-(--composer-control-size) min-w-0 shrink items-center gap-1.5 rounded-md px-2 text-xs text-(--ui-text-secondary)"
          data-mode={mode}
          data-slot="profile-pill"
          role="group"
        >
          {glyph}
          <span className="truncate">{visibleOwner}</span>
        </span>
      </Tip>
    )
  }

  return (
    <DropdownMenu>
      <Tip label={visibleOwner} placement="control">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${t.profiles.title}: ${visibleOwner}`}
            className={PILL}
            data-mode={mode}
            data-slot="profile-pill"
            type="button"
            variant="ghost"
          >
            {glyph}
            <span className="truncate">{visibleOwner}</span>
            <ChevronDown aria-hidden="true" className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent
        align="start"
        className="min-w-52 max-w-72"
        collisionPadding={8}
        onCloseAutoFocus={() => releaseTypingFocus()}
        side="top"
        sideOffset={8}
      >
        <DropdownMenuLabel>
          {menuConnectionLabel ? `${t.profiles.title} · ${menuConnectionLabel}` : t.profiles.title}
        </DropdownMenuLabel>
        <DropdownMenuRadioGroup onValueChange={selectProfile} value={menuValue}>
          {orderedProfiles.map(item => (
            <ProfileOption color={resolveProfileColor(item.name, colors)} key={item.name} profile={item} />
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function sortProfiles(profiles: ProfileInfo[], order: string[]): ProfileInfo[] {
  const defaultProfile = profiles.find(profile => profile.is_default)

  const named = sortByProfileOrder(
    profiles.filter(profile => !profile.is_default),
    order
  )

  return defaultProfile ? [defaultProfile, ...named] : named
}

function ProfileOption({ color, profile }: { color: null | string; profile: ProfileInfo }) {
  const label = profileLabel(profile)

  return (
    <DropdownMenuRadioItem className="min-w-0" value={profile.name}>
      <span className="flex min-w-0 items-center gap-1.5">
        <ProfileGlyph aria-hidden="true" color={color} isDefault={profile.is_default} name={profile.name} />
        <span className="min-w-0 flex-1 truncate">{label}</span>
        {label !== profile.name && <span className="shrink-0 text-xs text-(--ui-text-tertiary)">{profile.name}</span>}
      </span>
    </DropdownMenuRadioItem>
  )
}
