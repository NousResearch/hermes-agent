import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { DESKTOP_LANGUAGES, useI18n, useTranslation } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { SettingsContent } from './primitives'

const themeDescriptionKey = (name: string) => `settings.appearance.theme.${name}.description`

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

function SectionHead({ title, description, control }: { title: string; description: string; control?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <div className="min-w-0">
        <div className="text-[length:var(--conversation-text-font-size)] font-medium">{title}</div>
        <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </div>
      </div>
      {control && <div className="shrink-0">{control}</div>}
    </div>
  )
}

export function AppearanceSettings() {
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const { isSavingLanguage, language, setLanguage } = useI18n()
  const t = useTranslation()
  const toolViewMode = useStore($toolViewMode)

  const languageOptions = DESKTOP_LANGUAGES.map(option => ({
    id: option.id,
    label: t(option.translationKey)
  }))

  const modeOptions = MODE_OPTIONS.map(option => ({
    id: option.id,
    icon: option.icon,
    label: t(option.labelKey)
  }))

  const selectLanguage = async (nextLanguage: typeof language) => {
    if (nextLanguage === language || isSavingLanguage) {
      return
    }

    triggerHaptic('selection')

    try {
      await setLanguage(nextLanguage)
      triggerHaptic('success')
    } catch (error) {
      notifyError(error, t('settings.appearance.language.saveError'))
    }
  }

  return (
    <SettingsContent>
      <div className="grid gap-8">
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {t('settings.appearance.description')}
        </p>

        <section>
          <SectionHead
            control={
              <div className="grid justify-items-end gap-1.5">
                <SegmentedControl
                  onChange={next => void selectLanguage(next)}
                  options={languageOptions}
                  value={language}
                />
                {isSavingLanguage && (
                  <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {t('settings.appearance.language.saving')}
                  </span>
                )}
              </div>
            }
            description={t('settings.appearance.language.description')}
            title={t('settings.appearance.language.title')}
          />
        </section>

        <section>
          <SectionHead
            control={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('crisp')
                  setMode(id)
                }}
                options={modeOptions}
                value={mode}
              />
            }
            description={t('settings.appearance.mode.description')}
            title={t('settings.appearance.mode.title')}
          />
        </section>

        <section>
          <SectionHead
            control={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setToolViewMode(id)
                }}
                options={
                  [
                    { id: 'product', label: t('settings.appearance.toolDisplay.product') },
                    { id: 'technical', label: t('settings.appearance.toolDisplay.technical') }
                  ] as const
                }
                value={toolViewMode}
              />
            }
            description={t('settings.appearance.toolDisplay.description')}
            title={t('settings.appearance.toolDisplay.title')}
          />
        </section>

        <section className="grid gap-3">
          <SectionHead
            description={t('settings.appearance.theme.description')}
            title={t('settings.appearance.theme.title')}
          />
          <div className="grid gap-x-4 gap-y-5 sm:grid-cols-2 xl:grid-cols-3">
            {availableThemes.map(theme => {
              const active = themeName === theme.name

              return (
                <button
                  className="group text-left"
                  key={theme.name}
                  onClick={() => {
                    triggerHaptic('crisp')
                    setTheme(theme.name)
                  }}
                  type="button"
                >
                  <div
                    className={cn(
                      'rounded-xl transition',
                      active
                        ? 'ring-2 ring-primary ring-offset-2 ring-offset-background'
                        : 'opacity-90 group-hover:opacity-100'
                    )}
                  >
                    <ThemePreview name={theme.name} />
                  </div>
                  <div className="mt-2.5 flex items-start justify-between gap-2 px-0.5">
                    <div className="min-w-0">
                      <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
                        {theme.label}
                      </div>
                      <div className="mt-0.5 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                        {t(themeDescriptionKey(theme.name))}
                      </div>
                    </div>
                    {active && <Check className="mt-0.5 size-4 shrink-0 text-primary" />}
                  </div>
                </button>
              )
            })}
          </div>
        </section>
      </div>
    </SettingsContent>
  )
}
