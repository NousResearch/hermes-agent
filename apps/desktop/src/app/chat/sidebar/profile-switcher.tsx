import { useStore } from '@nanostores/react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { profileColor, profileColorSoft } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import {
  $activeGatewayProfile,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey,
  refreshActiveProfile,
  selectProfile,
  setShowAllProfiles
} from '@/store/profile'

import { PROFILES_ROUTE } from '../../routes'

// Arc-Spaces-style profile rail at the sidebar foot: a default↔all toggle pinned
// left, the colored named profiles scrolling between, and Manage pinned right.
// The active profile pops in its own color — the "where am I" cue. Only mounted
// when >1 profile exists, so single-profile users never see it.
export function ProfileRail() {
  const profiles = useStore($profiles)
  const scope = useStore($profileScope)
  const gatewayProfile = useStore($activeGatewayProfile)
  const navigate = useNavigate()

  const isAll = scope === ALL_PROFILES
  const activeKey = normalizeProfileKey(gatewayProfile)
  const defaultProfile = profiles.find(profile => profile.is_default)
  const onDefault = !isAll && activeKey === 'default'

  const named = profiles.filter(profile => !profile.is_default).sort((a, b) => a.name.localeCompare(b.name))

  // Re-pull the running profile + list on mount so a profile created elsewhere
  // shows up; cheap and best-effort.
  useEffect(() => {
    void refreshActiveProfile()
  }, [])

  return (
    <div aria-label="Profiles" className="flex items-center gap-0.5" role="tablist">
      {/* One button toggles default ↔ all: home face when scoped to a profile,
          layers face when showing everything. Pinned left like Manage is right. */}
      {defaultProfile ? (
        // On default → toggle to all. Anywhere else (all view or a named
        // profile) → return to default. So leaving a profile never lands on all.
        <ProfilePill
          active={isAll || onDefault}
          glyph={isAll ? 'layers' : 'home'}
          label={onDefault ? 'Show all profiles' : `Switch to ${defaultProfile.name}`}
          onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
        />
      ) : (
        <ProfilePill active={isAll} glyph="layers" label="All profiles" onSelect={() => setShowAllProfiles(true)} />
      )}

      <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {named.map(profile => (
          <ProfileSquare
            active={!isAll && normalizeProfileKey(profile.name) === activeKey}
            color={profileColor(profile.name)}
            key={profile.name}
            label={profile.name}
            onSelect={() => selectProfile(profile.name)}
          />
        ))}
      </div>

      <ProfilePill active={false} glyph="ellipsis" label="Manage profiles…" onSelect={() => navigate(PROFILES_ROUTE)} />
    </div>
  )
}

interface ProfilePillProps {
  active: boolean
  // home / All / Manage are glyph action buttons (navigation, not identity).
  glyph: string
  label: string
  onSelect: () => void
}

function ProfilePill({ active, glyph, label, onSelect }: ProfilePillProps) {
  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground'
        )}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name={glyph} size="0.875rem" />
      </Button>
    </Tip>
  )
}

interface ProfileSquareProps {
  active: boolean
  color: null | string
  label: string
  onSelect: () => void
}

// A profile *is* its colored square — no icon-button chrome. Soft profile-tint
// fill + the initial in the full color; the active one pops to full opacity with
// a color ring. These pack tightly so the rail reads as a strip of profiles.
function ProfileSquare({ active, color, label, onSelect }: ProfileSquareProps) {
  const hue = color ?? 'var(--ui-text-quaternary)'

  return (
    <Tip label={label}>
      <button
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'grid size-5 shrink-0 place-items-center rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none transition-opacity hover:opacity-100',
          active ? 'opacity-100' : 'opacity-55'
        )}
        onClick={onSelect}
        style={{
          backgroundColor: profileColorSoft(hue, active ? 30 : 22),
          boxShadow: active ? `inset 0 0 0 1.5px ${hue}` : undefined,
          color: color ?? undefined
        }}
        type="button"
      >
        {label.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
      </button>
    </Tip>
  )
}
