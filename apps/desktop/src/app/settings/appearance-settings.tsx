import { useStore } from '@nanostores/react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { ListRow, SectionHeading, SettingsContent } from './primitives'

function ThemePreview({ name }: { name: string }) {
  const t = BUILTIN_THEMES[name]

  if (!t) {
    return null
  }

  const c = t.colors

  return (
    <div
      className="h-20 overflow-hidden rounded-xl border shadow-xs"
      style={{ backgroundColor: c.background, borderColor: c.border }}
    >
      <div className="flex h-full">
        <div
          className="w-12 border-r"
          style={{
            backgroundColor: c.sidebarBackground ?? c.muted,
            borderColor: c.sidebarBorder ?? c.border
          }}
        />
        <div className="flex flex-1 flex-col gap-2 p-3">
          <div className="h-2.5 w-16 rounded-full" style={{ backgroundColor: c.foreground }} />
          <div className="h-2 w-24 rounded-full" style={{ backgroundColor: c.mutedForeground }} />
          <div className="mt-auto flex justify-end">
            <div
              className="h-5 w-16 rounded-full border"
              style={{
                backgroundColor: c.userBubble ?? c.muted,
                borderColor: c.userBubbleBorder ?? c.border
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export function AppearanceSettings() {
  const { t, isSavingLocale } = useI18n()
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const profiles = useStore($profiles)
  const activeProfileKey = normalizeProfileKey(useStore($activeGatewayProfile))
  const a = t.settings.appearance

  // Themes save per profile. Surface that only when the user actually has more
  // than one profile (single-profile installs never see the distinction).
  const showProfileNote = profiles.length > 1

  const activeProfileName =
    profiles.find(profile => normalizeProfileKey(profile.name) === activeProfileKey)?.name ?? activeProfileKey

  const modeOptions = MODE_OPTIONS.map(({ id, icon }) => ({ icon, id, label: t.settings.modeOptions[id].label }))

  const toolOptions = [
    { id: 'product', label: a.product },
    { id: 'technical', label: a.technical }
  ] as const

  return (
    <SettingsContent>
      <div>
        <SectionHeading icon={Palette} title={a.title} />
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {a.intro}
        </p>

        <div className="mt-2 divide-y divide-(--ui-stroke-tertiary)">
          <ListRow
            action={<LanguageSwitcher />}
            description={isSavingLocale ? t.language.saving : t.language.description}
            title={t.language.label}
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('crisp')
                  setMode(id)
                }}
                options={modeOptions}
                value={mode}
              />
            }
            description={a.colorModeDesc}
            title={a.colorMode}
          />

          <ListRow
            below={
              <>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {availableThemes.map(theme => {
                    const active = themeName === theme.name

                    return (
                      <button
                        className={cn(
                          'rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2 text-left transition hover:bg-(--chrome-action-hover)',
                          active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                        )}
                        key={theme.name}
                        onClick={() => {
                          triggerHaptic('crisp')
                          setTheme(theme.name)
                        }}
                        type="button"
                      >
                        <ThemePreview name={theme.name} />
                        <div className="mt-3 flex items-start justify-between gap-3 px-1">
                          <div className="min-w-0">
                            <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
                              {theme.label}
                            </div>
                            <div className="mt-0.5 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                              {theme.description}
                            </div>
                          </div>
                          {active && (
                            <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
                              <Check className="size-3.5" />
                            </span>
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
                {showProfileNote && (
                  <p className="mt-3 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {a.themeProfileNote(activeProfileName)}
                  </p>
                )}
              </>
            }
            description={a.themeDesc}
            title={a.themeTitle}
            wide
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setToolViewMode(id)
                }}
                options={toolOptions}
                value={toolViewMode}
              />
            }
            description={a.toolViewDesc}
            title={a.toolViewTitle}
          />
        </div>
      </div>
    </SettingsContent>
  )
}
