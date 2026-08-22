import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, Users } from '@/lib/icons'
import { $profiles, normalizeProfileKey, profileLabel, refreshProfiles } from '@/store/profile'
import { $starmapSelectedProfiles, loadStarmapGraph, toggleStarmapProfile } from '@/store/starmap'
import type { ProfileInfo } from '@/types/hermes'

interface ProfileSelectorProps {
  /** Profile the gateway is currently on — the "primary" profile. */
  activeProfile: string
}

/** Checkbox selector for bots to include in the multi-profile star map.
 *  Lives in the upper-left corner of the starmap view. */
export function ProfileSelector({ activeProfile }: ProfileSelectorProps) {
  const { t } = useI18n()
  const profiles = useStore($profiles)
  const selectedProfiles = useStore($starmapSelectedProfiles)
  const [open, setOpen] = useState(false)

  // Ensure profiles are loaded on mount
  useEffect(() => {
    if (profiles.length === 0) {
      void refreshProfiles()
    }
  }, [profiles.length])

  // Sort profiles: active first, then by name
  const sortedProfiles = useMemo(() => {
    const activeKey = normalizeProfileKey(activeProfile)
    return [...profiles].sort((a, b) => {
      const aActive = normalizeProfileKey(a.name) === activeKey
      const bActive = normalizeProfileKey(b.name) === activeKey
      if (aActive && !bActive) return -1
      if (!aActive && bActive) return 1
      return profileLabel(a).localeCompare(profileLabel(b))
    })
  }, [profiles, activeProfile])

  // On first load, initialize selection to the active profile if empty
  useEffect(() => {
    if (selectedProfiles.length === 0 && activeProfile) {
      toggleStarmapProfile(normalizeProfileKey(activeProfile), true)
    }
  }, [activeProfile, selectedProfiles.length])

  // Reload graph when selection changes
  useEffect(() => {
    if (selectedProfiles.length > 0) {
      void loadStarmapGraph(true)
    }
  }, [selectedProfiles])

  const handleToggle = (profile: ProfileInfo, enabled: boolean) => {
    toggleStarmapProfile(normalizeProfileKey(profile.name), enabled)
  }

  const handleSelectAll = () => {
    profiles.forEach(p => toggleStarmapProfile(normalizeProfileKey(p.name), true))
  }

  const handleSelectNone = () => {
    // Always keep at least the active profile selected
    profiles.forEach(p => {
      const key = normalizeProfileKey(p.name)
      if (key !== normalizeProfileKey(activeProfile)) {
        toggleStarmapProfile(key, false)
      }
    })
  }

  // Don't show the selector if there's only one profile
  if (profiles.length <= 1) {
    return null
  }

  const selectedCount = selectedProfiles.length
  const label = selectedCount === 0
    ? t.starmap.profileNone
    : selectedCount === profiles.length
      ? t.starmap.profileAll
      : t.starmap.profileCount(selectedCount)

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <Tip label={t.starmap.profileSelectorHint}>
        <PopoverTrigger asChild>
          <Button
            className="gap-1.5 text-xs"
            size="sm"
            variant="ghost"
          >
            <Users className="size-3.5" />
            <span className="hidden sm:inline">{label}</span>
            <ChevronDown className="size-3" />
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent
        align="start"
        className="w-56 p-1"
        side="bottom"
        sideOffset={4}
      >
        <div className="mb-1 flex items-center justify-between px-2 py-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t.starmap.profileSelector}
          </span>
          <div className="flex gap-1">
            <button
              className="cursor-pointer text-[0.65rem] text-muted-foreground/70 hover:text-foreground"
              onClick={handleSelectAll}
              type="button"
            >
              All
            </button>
            <span className="text-muted-foreground/40">|</span>
            <button
              className="cursor-pointer text-[0.65rem] text-muted-foreground/70 hover:text-foreground"
              onClick={handleSelectNone}
              type="button"
            >
              None
            </button>
          </div>
        </div>
        <div className="max-h-64 overflow-y-auto">
          {sortedProfiles.map(profile => {
            const key = normalizeProfileKey(profile.name)
            const isSelected = selectedProfiles.includes(key)
            const isActive = key === normalizeProfileKey(activeProfile)

            return (
              <button
                className="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-(--ui-control-active-background)"
                key={key}
                onClick={() => handleToggle(profile, !isSelected)}
                type="button"
              >
                <div
                  className={`flex size-4 shrink-0 items-center justify-center rounded border ${
                    isSelected
                      ? 'border-primary bg-primary text-primary-foreground'
                      : 'border-muted-foreground/30'
                  }`}
                >
                  {isSelected && <Check className="size-3" />}
                </div>
                <span className="min-w-0 flex-1 truncate">
                  {profileLabel(profile)}
                  {isActive && (
                    <span className="ml-1 text-[0.6rem] text-muted-foreground/70">
                      (current)
                    </span>
                  )}
                </span>
                {profile.display_name && profile.display_name !== profile.name && (
                  <span className="shrink-0 text-[0.6rem] text-muted-foreground/50">
                    {profile.name}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
