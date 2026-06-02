import { useStore } from '@nanostores/react'
import { useTranslation } from 'react-i18next'

import { triggerHaptic } from '@/lib/haptics'
import { LANGUAGE_LABELS, SUPPORTED_LANGUAGES, type SupportedLanguage } from '@/lib/i18n'
import { Check, Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { Pill, SectionHeading, SettingsContent } from './primitives'

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
  const { t, i18n } = useTranslation()
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const activeTheme = availableThemes.find(t => t.name === themeName)

  const currentLanguage: SupportedLanguage =
    SUPPORTED_LANGUAGES.find(lang => lang === (i18n.resolvedLanguage ?? i18n.language)) ?? 'en'

  return (
    <SettingsContent>
      <div className="space-y-5">
        <div>
          <SectionHeading icon={Palette} title={t('settings:appearance.title')} />
          <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {t('settings:appearance.description')}
          </p>
        </div>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings:appearance.language.title')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings:appearance.language.description')}
              </div>
            </div>
            <Pill>{LANGUAGE_LABELS[currentLanguage]}</Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {SUPPORTED_LANGUAGES.map(lang => {
              const active = currentLanguage === lang

              return (
                <button
                  className={cn(
                    'group rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2.5 text-left transition hover:bg-(--chrome-action-hover)',
                    active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                  )}
                  key={lang}
                  onClick={() => {
                    triggerHaptic('crisp')
                    void i18n.changeLanguage(lang)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-[length:var(--conversation-text-font-size)] font-medium">
                      {LANGUAGE_LABELS[lang]}
                    </div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {lang}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings:appearance.colorMode')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings:appearance.colorModeDescription')}
              </div>
            </div>
            <Pill>
              {t(`settings:appearance.mode.${mode}`)}
            </Pill>
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
              <div className="text-sm font-medium">{t('settings:appearance.toolCallDisplay')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {t('settings:appearance.toolCallDisplayDescription')}
              </div>
            </div>
            <Pill>
              {t(
                toolViewMode === 'technical'
                  ? 'settings:appearance.toolView.technical'
                  : 'settings:appearance.toolView.product',
                toolViewMode === 'technical' ? 'Technical' : 'Product'
              )}
            </Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {(
              [
                {
                  id: 'product' as const,
                  key: 'product',
                  fallbackLabel: 'Product',
                  fallbackDescription: 'Human-friendly tool activity with concise summaries.'
                },
                {
                  id: 'technical' as const,
                  key: 'technical',
                  fallbackLabel: 'Technical',
                  fallbackDescription: 'Include raw tool args/results and low-level details.'
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
                      {t(`settings:appearance.toolView.${option.key}`, option.fallbackLabel)}
                    </div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {t(`settings:appearance.toolView.${option.key}Description`, option.fallbackDescription)}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{t('settings:appearance.theme')}</div>
              <div className="mt-1 text-xs text-muted-foreground">{t('settings:appearance.themeDescription')}</div>
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
        </section>
      </div>
    </SettingsContent>
  )
}
