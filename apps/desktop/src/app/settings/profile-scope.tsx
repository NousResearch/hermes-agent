import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile, $profiles, normalizeProfileKey, refreshProfiles } from '@/store/profile'
import { $settingsScopeOverride, setSettingsScope } from '@/store/settings-scope'

// The same chip affordance the Gateway page uses for its per-profile
// connection overrides (gateway-settings ScopeChip). That one stays local to
// gateway-settings — its `null` chip means "all profiles", while here every
// chip is a concrete profile whose config the page edits.
export function ScopeChip({ active, label, onSelect }: { active: boolean; label: string; onSelect: () => void }) {
  return (
    <button
      className={cn(
        'rounded-full border px-3 py-1 text-[length:var(--conversation-caption-font-size)] transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) text-(--ui-text-primary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover)'
      )}
      onClick={onSelect}
      type="button"
    >
      {label}
    </button>
  )
}

/** Shared "Applies to" profile selector for the config-backed settings pages
 *  (Model, Workspace, Safety, Memory & Context, Voice, Tools & Keys) and the
 *  Messaging overlay. Backed by one nanostore ($settingsScopeOverride) so the
 *  selection persists across pages. Hidden with fewer than two profiles, so
 *  single-profile users never see it and every request keeps its unscoped
 *  default shape. */
export function SettingsProfileScope({ className }: { className?: string }) {
  const { t } = useI18n()
  const scope = t.settings.profileScope
  const override = useStore($settingsScopeOverride)
  const active = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)

  // Refresh lazily so a profile created elsewhere shows up; the cached list
  // paints immediately. Best-effort — a failure keeps the cached roster.
  useEffect(() => {
    void refreshProfiles().catch(() => undefined)
  }, [])

  if (profiles.length < 2) {
    return null
  }

  const selected = normalizeProfileKey(override ?? active)
  const defaultProfile = profiles.find(profile => profile.is_default)
  const editingNonDefault = defaultProfile ? selected !== normalizeProfileKey(defaultProfile.name) : false

  return (
    <div className={cn('grid gap-2', className)}>
      <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
        {scope.appliesTo}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {profiles.map(profile => (
          <ScopeChip
            active={normalizeProfileKey(profile.name) === selected}
            key={profile.name}
            label={profile.name}
            onSelect={() => setSettingsScope(profile.name)}
          />
        ))}
      </div>
      {/* The scope follows the app's ACTIVE profile when no override is set —
          which, after opening any Bot Mode chat, is the BOT's profile. Users
          reasonably assume Settings edit their main config, so an edit landing
          in profiles/<bot>/config.yaml with only a faint chip tint as the tell
          is a silent misdirect (the #89190/#89162 report class). Always state
          the target when it isn't the default profile, and make it loud. */}
      {override !== null || editingNonDefault ? (
        <p
          className={cn(
            'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)',
            editingNonDefault ? 'font-medium text-(--ui-accent)' : 'text-(--ui-text-tertiary)'
          )}
        >
          {scope.editsProfile(selected)}
        </p>
      ) : null}
    </div>
  )
}
