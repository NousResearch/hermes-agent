import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import type { DesktopMarketplaceSearchItem } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Download, Loader2, Palette, Trash2 } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { $embedAllowed, $embedMode, clearEmbedAllowed, type EmbedMode, setEmbedMode } from '@/store/embed-consent'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { $translucency, setTranslucency } from '@/store/translucency'
import { getBaseColors, useTheme } from '@/themes/context'
import { installVscodeThemeFromMarketplace } from '@/themes/install'
import type { DesktopTheme } from '@/themes/types'
import { $marketplaceInstalls, isUserTheme, removeUserTheme } from '@/themes/user-themes'

import { MODE_OPTIONS } from './constants'
import { PetSettings } from './pet-settings'
import { ListRow, SectionHeading, SettingsContent } from './primitives'

function ThemePreview({ name, mode }: { name: string; mode: 'light' | 'dark' }) {
  // Preview in the *current* mode: the dark palette in Dark, and the light
  // palette in Light — synthesizing one for dark-only themes — so every card
  // tracks the Light/Dark toggle, exactly like the app itself does.
  const c = getBaseColors(name, mode)

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

function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delayMs)

    return () => clearTimeout(handle)
  }, [value, delayMs])

  return debounced
}

const compactNumber = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })

/**
 * Live VS Code Marketplace theme search (the same backend as the Cmd-K "Install
 * theme…" page). Renders below the local grid when there's a query: each row
 * downloads + converts + installs via `installVscodeThemeFromMarketplace` and
 * activates it. Extensions already imported locally are marked installed.
 */
function MarketplaceThemeResults({
  query,
  installs,
  onInstalled
}: {
  query: string
  installs: ReadonlyMap<string, DesktopTheme>
  onInstalled: (name: string) => void
}) {
  const { t } = useI18n()
  const copy = t.commandCenter.installTheme
  const debounced = useDebounced(query.trim(), 300)
  const [installingId, setInstallingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const search = useQuery({
    enabled: debounced.length > 0,
    queryFn: () => window.hermesDesktop?.themes?.searchMarketplace(debounced) ?? Promise.resolve([]),
    queryKey: ['marketplace-themes-settings', debounced],
    staleTime: 5 * 60 * 1000
  })

  // Already installed → just re-activate it; never re-download what we have.
  const select = (item: DesktopMarketplaceSearchItem) => {
    const owned = installs.get(item.extensionId)

    if (owned) {
      triggerHaptic('crisp')
      onInstalled(owned.name)

      return
    }

    void install(item)
  }

  const install = async (item: DesktopMarketplaceSearchItem) => {
    if (installingId) {
      return
    }

    setInstallingId(item.extensionId)
    setError(null)

    try {
      const theme = await installVscodeThemeFromMarketplace(item.extensionId)

      triggerHaptic('crisp')
      onInstalled(theme.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : copy.error)
    } finally {
      setInstallingId(null)
    }
  }

  if (!debounced) {
    return null
  }

  const header = (
    <p className="mb-2 mt-4 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
      From the VS Code Marketplace
    </p>
  )

  if (search.isLoading) {
    return (
      <>
        {header}
        <p className="flex items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <Loader2 className="size-3.5 animate-spin" />
          {copy.loading}
        </p>
      </>
    )
  }

  if (search.isError) {
    return (
      <>
        {header}
        <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-red)">{copy.error}</p>
      </>
    )
  }

  const results = search.data ?? []

  if (results.length === 0) {
    return (
      <>
        {header}
        <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{copy.empty}</p>
      </>
    )
  }

  return (
    <>
      {header}
      {error && <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-red)">{error}</p>}
      <div className="grid gap-2 sm:grid-cols-2">
        {results.map(item => {
          const busy = installingId === item.extensionId
          const done = installs.has(item.extensionId)

          return (
            <button
              className={cn(
                'flex items-center gap-2.5 px-2.5 py-2 text-left disabled:opacity-60',
                selectableCardClass({ prominent: done })
              )}
              disabled={Boolean(installingId) && !busy}
              key={item.extensionId}
              onClick={() => select(item)}
              type="button"
            >
              <Palette className="size-4 shrink-0 text-(--ui-text-tertiary)" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[length:var(--conversation-text-font-size)] font-medium">
                  {item.displayName}
                </span>
                <span className="block truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {item.publisher}
                  {item.installs > 0 ? ` · ${copy.installs(compactNumber.format(item.installs))}` : ''}
                </span>
              </span>
              <span className="shrink-0 text-(--ui-text-tertiary)">
                {busy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : done ? (
                  <Check className="size-4 text-(--ui-green)" />
                ) : (
                  <Download className="size-4" />
                )}
              </span>
            </button>
          )
        })}
      </div>
    </>
  )
}

export function AppearanceSettings() {
  const { t, isSavingLocale } = useI18n()
  const { themeName, mode, resolvedMode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const embedMode = useStore($embedMode)
  const embedAllowed = useStore($embedAllowed)
  const translucency = useStore($translucency)
  const installs = useStore($marketplaceInstalls)
  const a = t.settings.appearance

  const [query, setQuery] = useState('')

  // One box does double duty: filter installed themes live (below), and run a
  // name search against the VS Code Marketplace (the Cmd-K "Install theme…"
  // backend) for anything not already installed.
  const needle = query.trim().toLowerCase()

  const filteredThemes = availableThemes
    .filter(
      theme =>
        !needle ||
        theme.label.toLowerCase().includes(needle) ||
        theme.name.toLowerCase().includes(needle) ||
        theme.description.toLowerCase().includes(needle)
    )
    // Active theme first; stable sort keeps the rest in their original order.
    .sort((a, b) => Number(b.name === themeName) - Number(a.name === themeName))

  const modeOptions = MODE_OPTIONS.map(({ id, icon }) => ({ icon, id, label: t.settings.modeOptions[id].label }))

  const toolOptions = [
    { id: 'product', label: a.product },
    { id: 'technical', label: a.technical }
  ] as const

  const embedOptions = [
    { id: 'ask', label: a.embedsAsk },
    { id: 'always', label: a.embedsAlways },
    { id: 'off', label: a.embedsOff }
  ] as const satisfies readonly { id: EmbedMode; label: string }[]

  return (
    <SettingsContent>
      <div>
        <SectionHeading icon={Palette} title={a.title} />
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {a.intro}
        </p>

        <div className="mt-2">
          <ListRow
            action={<LanguageSwitcher />}
            description={isSavingLocale ? t.language.saving : t.language.description}
            title={t.language.label}
          />

          <ListRow
            below={
              <>
                {/* One search box: filters your installed themes (the grid)
                    and live-searches the VS Code Marketplace below. */}
                <div className="mt-3">
                  <input
                    className="w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 text-[length:var(--conversation-caption-font-size)] outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-secondary)"
                    onChange={event => setQuery(event.target.value)}
                    placeholder="Search your themes or the VS Code Marketplace…"
                    spellCheck={false}
                    value={query}
                  />
                </div>

                {/* Fixed-height scroll area so the (growing) theme list never
                    runs the page long; the grid scrolls inside it. */}
                <div className="mt-3 max-h-96 overflow-y-auto pr-1">
                  {filteredThemes.length === 0 ? (
                    needle ? (
                      <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                        No installed themes match "{query.trim()}".
                      </p>
                    ) : null
                  ) : (
                    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                      {filteredThemes.map(theme => {
                        const active = themeName === theme.name
                        const removable = isUserTheme(theme.name)

                        return (
                          <div className="group relative" key={theme.name}>
                            <button
                              className={cn('w-full p-2 text-left', selectableCardClass({ active, prominent: true }))}
                              onClick={() => {
                                triggerHaptic('crisp')
                                setTheme(theme.name)
                              }}
                              type="button"
                            >
                              <ThemePreview mode={resolvedMode} name={theme.name} />
                              <div className="mt-3 px-1">
                                <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
                                  {theme.label}
                                </div>
                                <div className="mt-0.5 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                                  {theme.description}
                                </div>
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
                  )}
                  <MarketplaceThemeResults installs={installs} onInstalled={name => setTheme(name)} query={query} />
                </div>
              </>
            }
            description={a.themeDesc}
            title={
              <div className="flex items-center justify-between gap-3">
                <span>{a.themeTitle}</span>
                <SegmentedControl
                  onChange={id => {
                    triggerHaptic('crisp')
                    setMode(id)
                  }}
                  options={modeOptions}
                  value={mode}
                />
              </div>
            }
            wide
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
              <div className="flex flex-col items-end gap-1.5">
                <SegmentedControl
                  onChange={id => {
                    triggerHaptic('selection')
                    setEmbedMode(id)
                  }}
                  options={embedOptions}
                  value={embedMode}
                />
                {embedAllowed.length > 0 && (
                  <Button
                    onClick={() => {
                      triggerHaptic('selection')
                      clearEmbedAllowed()
                    }}
                    size="inline"
                    variant="text"
                  >
                    {a.embedsReset(embedAllowed.length)}
                  </Button>
                )}
              </div>
            }
            description={a.embedsDesc}
            title={a.embedsTitle}
          />
        </div>
      </div>

      <div className="mt-6">
        <PetSettings />
      </div>
    </SettingsContent>
  )
}
