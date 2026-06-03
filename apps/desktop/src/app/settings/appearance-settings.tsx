import { useStore } from '@nanostores/react'

import { DESKTOP_LANGUAGES, useI18n, useTranslation } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { Pill, SectionHeading, SettingsContent } from './primitives'

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

export function AppearanceSettings() {
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const { isSavingLanguage, language, setLanguage } = useI18n()
  const t = useTranslation()
  const toolViewMode = useStore($toolViewMode)
  const activeTheme = availableThemes.find(t => t.name === themeName)
  const activeLanguage = DESKTOP_LANGUAGES.find(item => item.id === language)
  const activeMode = MODE_OPTIONS.find(option => option.id === mode)

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
      <div className="space-y-5">
        <div>
          <SectionHeading icon={Palette} title={t('settings.appearance.title')} />
          <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {t('settings.appearance.description')}
          </p>
        </div>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings.appearance.language.title')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings.appearance.language.description')}
              </div>
            </div>
            {activeLanguage && <Pill>{t(activeLanguage.translationKey)}</Pill>}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {DESKTOP_LANGUAGES.map(option => {
              const active = language === option.id

              return (
                <button
                  className={cn(
                    'group rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2.5 text-left transition hover:bg-(--chrome-action-hover) disabled:cursor-not-allowed disabled:opacity-60',
                    active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                  )}
                  disabled={isSavingLanguage}
                  key={option.id}
                  onClick={() => void selectLanguage(option.id)}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[length:var(--conversation-text-font-size)] font-medium">
                        {t(option.translationKey)}
                      </div>
                      <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                        {option.nativeLabel}
                      </div>
                    </div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
          {isSavingLanguage && (
            <div className="mt-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {t('settings.appearance.language.saving')}
            </div>
          )}
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings.appearance.mode.title')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings.appearance.mode.description')}
              </div>
            </div>
            {activeMode && <Pill>{t(activeMode.labelKey)}</Pill>}
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {MODE_OPTIONS.map(({ id, labelKey, descriptionKey, icon: Icon }) => {
              const active = mode === id

              return (
                <button
                  className={cn(
                    'group rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2.5 text-left transition hover:bg-(--chrome-action-hover)',
                    active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                  )}
                  key={id}
                  onClick={() => {
                    triggerHaptic('crisp')
                    setMode(id)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <span className="flex size-9 items-center justify-center rounded-lg bg-muted text-foreground transition group-hover:bg-background">
                      <Icon className="size-4" />
                    </span>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-2 text-[length:var(--conversation-text-font-size)] font-medium">
                    {t(labelKey)}
                  </div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {t(descriptionKey)}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings.appearance.toolDisplay.title')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings.appearance.toolDisplay.description')}
              </div>
            </div>
            <Pill>
              {toolViewMode === 'technical'
                ? t('settings.appearance.toolDisplay.technical')
                : t('settings.appearance.toolDisplay.product')}
            </Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {(
              [
                {
                  id: 'product',
                  labelKey: 'settings.appearance.toolDisplay.product',
                  descriptionKey: 'settings.appearance.toolDisplay.productDescription'
                },
                {
                  id: 'technical',
                  labelKey: 'settings.appearance.toolDisplay.technical',
                  descriptionKey: 'settings.appearance.toolDisplay.technicalDescription'
                }
              ] as const
            ).map(option => {
              const active = toolViewMode === option.id

              return (
                <button
                  className={cn(
                    'group rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2.5 text-left transition hover:bg-(--chrome-action-hover)',
                    active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                  )}
                  key={option.id}
                  onClick={() => {
                    triggerHaptic('selection')
                    setToolViewMode(option.id)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-[length:var(--conversation-text-font-size)] font-medium">
                      {t(option.labelKey)}
                    </div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {t(option.descriptionKey)}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings.appearance.theme.title')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings.appearance.theme.description')}
              </div>
            </div>
            {activeTheme && <Pill>{activeTheme.label}</Pill>}
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
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
                        {t(themeDescriptionKey(theme.name))}
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
        </section>
      </div>
    </SettingsContent>
  )
}
