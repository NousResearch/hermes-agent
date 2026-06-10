import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { Check, Download, Loader2, Palette, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'
import { useTranslation, LANGUAGE_LABELS } from '@/hooks/use-translation'

import { MODE_OPTIONS } from './constants'
import { Pill, SectionHeading, SettingsContent } from './primitives'

function ThemePreview({ name }: { name: string }) {
  const t = resolveTheme(name)

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

function VscodeThemeInstaller() {
  const { t } = useI18n()
  const { setTheme } = useTheme()
  const a = t.settings.appearance
  const [id, setId] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<{ kind: 'error' | 'success'; text: string } | null>(null)

  const install = async () => {
    const trimmed = id.trim()

    if (!trimmed || busy) {
      return
    }

    setBusy(true)
    setStatus(null)

    try {
      const theme = await installVscodeThemeFromMarketplace(trimmed)

      triggerHaptic('crisp')
      setTheme(theme.name)
      setStatus({ kind: 'success', text: a.installed(theme.label) })
      setId('')
    } catch (error) {
      setStatus({ kind: 'error', text: error instanceof Error ? error.message : a.installError })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center gap-2">
        <input
          className="min-w-0 flex-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 font-mono text-[length:var(--conversation-caption-font-size)] outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-secondary)"
          disabled={busy}
          onChange={event => {
            setId(event.target.value)
            setStatus(null)
          }}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              void install()
            }
          }}
          placeholder={a.installPlaceholder}
          spellCheck={false}
          value={id}
        />
        <button
          className="inline-flex items-center gap-1.5 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) px-3 py-1.5 text-[length:var(--conversation-caption-font-size)] font-medium transition hover:bg-(--chrome-action-hover) disabled:opacity-50"
          disabled={busy || !id.trim()}
          onClick={() => void install()}
          type="button"
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
          {busy ? a.installing : a.installButton}
        </button>
      </div>
      {status && (
        <p
          className={cn(
            'mt-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)',
            status.kind === 'error' ? 'text-(--ui-red)' : 'text-(--ui-text-tertiary)'
          )}
        >
          {status.text}
        </p>
      )}
    </div>
  )
}

export function AppearanceSettings() {
  const { t, isSavingLocale, locale, setLocale } = useI18n()
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const activeTheme = availableThemes.find(t => t.name === themeName)
  const { t: translate, locale, setLocale, availableLocales } = useTranslation()

  return (
    <SettingsContent>
      <div className="space-y-5">
        <div>
          <SectionHeading icon={Palette} title={translate('appearance.title')} />
          <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {translate('appearance.description')}
          </p>
        </div>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{translate('appearance.colorMode')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {translate('appearance.colorModeDesc')}
              </div>
            </div>
            <Pill>{translate(`appearance.mode.${mode}`)}</Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {MODE_OPTIONS.map(({ id, label, description, icon: Icon }) => {
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
                  <div className="mt-2 text-[length:var(--conversation-text-font-size)] font-medium">{translate(label)}</div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {translate(description)}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{translate('appearance.toolCallDisplay')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {translate('appearance.toolCallDisplayDesc')}
              </div>
            </div>
            <Pill>{toolViewMode === 'technical' ? translate('appearance.toolViewTechnical') : translate('appearance.toolViewProduct')}</Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {(
              [
                {
                  id: 'product',
                  label: translate('appearance.toolViewProduct'),
                  description: translate('appearance.toolViewProductDesc')
                },
                {
                  id: 'technical',
                  label: translate('appearance.toolViewTechnical'),
                  description: translate('appearance.toolViewTechnicalDesc')
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
                    <div className="text-[length:var(--conversation-text-font-size)] font-medium">{option.label}</div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {option.description}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{translate('appearance.theme')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {translate('appearance.themeDesc')}
              </div>
            </div>
            {activeTheme && <Pill>{translate(`appearance.theme.${activeTheme.name}`)}</Pill>}
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
                        {translate(`appearance.theme.${theme.name}`)}
                      </div>
                      <div className="mt-0.5 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                        {translate(`appearance.theme.${theme.name}Desc`)}
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

        {/* Language selector */}
        <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) p-3 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{translate('appearance.language')}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {translate('appearance.languageDesc')}
              </div>
            </div>
            <Pill>{LANGUAGE_LABELS[locale]}</Pill>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1.5">
            {availableLocales.map(code => (
              <button
                key={code}
                onClick={() => {
                  triggerHaptic('selection')
                  setLocale(code)
                }}
                className={cn(
                  'flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs transition-colors',
                  code === locale
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-(--ui-text-tertiary) hover:text-(--ui-text-primary) hover:bg-(--chrome-action-hover)'
                )}
                type="button"
              >
                {code === locale && <Check className="size-3" />}
                {LANGUAGE_LABELS[code] ?? code}
              </button>
            ))}
          </div>
        </section>
      </div>
    </SettingsContent>
  )
}
