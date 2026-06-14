import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { COMPLETION_SOUND_VARIANTS, previewCompletionSound } from '@/lib/completion-sound'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Download, Loader2, Palette, Play, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $completionSoundVariantId, setCompletionSoundVariantId } from '@/store/completion-sound'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { $translucency, setTranslucency } from '@/store/translucency'
import { useTheme } from '@/themes/context'
import { installVscodeThemeFromMarketplace } from '@/themes/install'
import { isUserTheme, removeUserTheme, resolveTheme } from '@/themes/user-themes'

import { CONTROL_TEXT, MODE_OPTIONS } from './constants'
import { ListRow, SectionHeading, SettingsContent } from './primitives'

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
  const { t, isSavingLocale } = useI18n()
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const completionSoundVariantId = useStore($completionSoundVariantId)
  const toolViewMode = useStore($toolViewMode)
  const translucency = useStore($translucency)
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
            action={
              <div className="flex items-center gap-3">
                <input
                  aria-label={a.translucencyTitle}
                  className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
                  max={100}
                  min={0}
                  onChange={event => {
                    triggerHaptic('selection')
                    setTranslucency(Number(event.target.value))
                  }}
                  step={5}
                  style={{ accentColor: 'var(--dt-primary)' }}
                  type="range"
                  value={translucency}
                />
                <span className="w-9 text-right text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)">
                  {translucency}%
                </span>
              </div>
            }
            description={a.translucencyDesc}
            title={a.translucencyTitle}
          />

          <ListRow
            below={
              <>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {availableThemes.map(theme => {
                    const active = themeName === theme.name
                    const removable = isUserTheme(theme.name)

                    return (
                      <div className="group relative" key={theme.name}>
                        <button
                          className={cn(
                            'w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2 text-left transition hover:bg-(--chrome-action-hover)',
                            active && 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
                          )}
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
                        {removable && (
                          <button
                            aria-label={a.removeTheme}
                            className="absolute right-1.5 top-1.5 grid size-6 place-items-center rounded-md bg-(--ui-bg-elevated)/80 text-(--ui-text-tertiary) opacity-0 backdrop-blur-sm transition hover:text-(--ui-red) focus-visible:opacity-100 group-hover:opacity-100"
                            onClick={() => {
                              triggerHaptic('crisp')
                              removeUserTheme(theme.name)

                              // Re-normalize off the now-missing skin → default.
                              if (active) {
                                setTheme(theme.name)
                              }
                            }}
                            title={a.removeTheme}
                            type="button"
                          >
                            <Trash2 className="size-3.5" />
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
                <VscodeThemeInstaller />
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

          <ListRow
            action={
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Select
                  onValueChange={value => {
                    const variantId = Number.parseInt(value, 10)

                    setCompletionSoundVariantId(variantId)
                    previewCompletionSound(variantId)
                    triggerHaptic('selection')
                  }}
                  value={String(completionSoundVariantId)}
                >
                  <SelectTrigger className={cn('min-w-56', CONTROL_TEXT)}>
                    <SelectValue />
                  </SelectTrigger>

                  <SelectContent>
                    {COMPLETION_SOUND_VARIANTS.map(variant => (
                      <SelectItem key={variant.id} value={String(variant.id)}>
                        {variant.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Button
                  className="gap-1.5"
                  onClick={() => {
                    previewCompletionSound()
                    triggerHaptic('crisp')
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Play className="size-3.5" />

                  {a.completionSoundPreview}
                </Button>
              </div>
            }
            description={a.completionSoundDesc}
            title={a.completionSoundTitle}
          />
        </div>
      </div>
    </SettingsContent>
  )
}
