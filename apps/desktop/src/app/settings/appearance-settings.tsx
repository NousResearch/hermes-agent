import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { useDebounced } from '@/app/hooks/use-debounced'
import { LanguageSwitcher } from '@/components/language-switcher'
import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import type { DesktopMarketplaceSearchItem } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Download, Loader2, Palette, Trash2 } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $embedAllowed, $embedMode, clearEmbedAllowed, type EmbedMode, setEmbedMode } from '@/store/embed-consent'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $reactionsEnabled, setReactionsEnabled } from '@/store/reactions-enabled'
import { $reasoningCollapsedByDefault, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $sessionListDensity, type SessionListDensity, setSessionListDensity } from '@/store/session-list-density'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import {
  $translucency,
  beginTranslucencyPeek,
  endTranslucencyPeek,
  GLASS_IS_WINDOWS,
  GLASS_SCOPES,
  GLASS_SUPPORTED,
  glassMaterialForPicker,
  glassMaterialsFor,
  pulseTranslucencyPeek,
  resetTranslucencyPeek,
  setTranslucency,
  setTranslucencyFade,
  setTranslucencyMaterial,
  setTranslucencyMode,
  setTranslucencyScope,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  TRANSLUCENCY_SUPPORTED
} from '@/store/translucency'
import { $zoomPercent, setZoomPercent } from '@/store/zoom'
import { getBaseColors, useTheme } from '@/themes/context'
import { installVscodeThemeFromMarketplace } from '@/themes/install'
import type { DesktopTheme } from '@/themes/types'
import { $marketplaceInstalls, isUserTheme, removeUserTheme } from '@/themes/user-themes'

import { MODE_OPTIONS } from './constants'
import { PetSettings } from './pet-settings'
import { ListRow, SectionHeading, SettingsContent, ToggleRow } from './primitives'
import { APPEARANCE_SETTING_IDS } from './settings-search'
import { TerminalFontSetting } from './terminal-font-setting'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

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

// UI scale presets, as zoom percentages. 100 is Chromium's actual-size
// baseline; the shipped default is the 90% preset. Ids double as the percent
// values sent to the main process. A Cmd/Ctrl +/- step landing between
// presets highlights nothing, and the row description keeps showing the
// exact current percent.
const UI_SCALE_PRESETS = ['90', '100', '110', '125', '150', '175'] as const
const APPEARANCE_SEARCH_TARGETS = new Set<string>(Object.values(APPEARANCE_SETTING_IDS))
const appearanceSettingElementId = (id: string) => `setting-field-${id}`

type UiScalePreset = (typeof UI_SCALE_PRESETS)[number]

function matchUiScalePreset(percent: number): UiScalePreset | null {
  return UI_SCALE_PRESETS.find(preset => Number(preset) === percent) ?? null
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

// Keys a range input treats as a step, so the peek can flash the live window
// for keyboard adjustment the way a pointer drag holds it open.
const SLIDER_STEP_KEYS = new Set([
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'End',
  'Home',
  'PageDown',
  'PageUp'
])

interface TranslucencySliderProps {
  label: string
  onChange: (value: number) => void
  value: number
}

/**
 * One 0–100 lever, used up to twice: Clear's window opacity, and under Glass
 * the tint plus an optional native fade.
 *
 * Peek while the hand is on it — the overlay (scrim + near-opaque card) ghosts
 * so the window behind IS the live preview. The pointer pair covers
 * mouse/touch drags; the keyboard path pulses per step instead, and blur ends
 * any residual hold.
 */
function TranslucencySlider({ label, onChange, value }: TranslucencySliderProps) {
  return (
    <>
      <input
        aria-label={label}
        className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
        max={TRANSLUCENCY_MAX}
        min={TRANSLUCENCY_MIN}
        onBlur={endTranslucencyPeek}
        onChange={event => {
          triggerHaptic('selection')
          onChange(Number(event.target.value))
        }}
        onKeyDown={event => {
          if (SLIDER_STEP_KEYS.has(event.key)) {
            pulseTranslucencyPeek()
          }
        }}
        onLostPointerCapture={endTranslucencyPeek}
        onPointerDown={beginTranslucencyPeek}
        onPointerUp={endTranslucencyPeek}
        step={TRANSLUCENCY_STEP}
        style={{ accentColor: 'var(--dt-primary)' }}
        type="range"
        value={value}
      />
      <span className="w-9 text-right text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)">
        {value}%
      </span>
    </>
  )
}

interface GlassRowProps {
  children: React.ReactNode
  label: string
}

/** A labelled control in the Glass sub-panel: tint, fade, frost, area. */
function GlassRow({ children, label }: GlassRowProps) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-12 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {label}
      </span>
      {children}
    </div>
  )
}

export function AppearanceSettings() {
  const { t, isSavingLocale } = useI18n()
  const { themeName, mode, resolvedMode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const reasoningCollapsedByDefault = useStore($reasoningCollapsedByDefault)
  const sessionListDensity = useStore($sessionListDensity)
  const zoomPercent = useStore($zoomPercent)
  const embedMode = useStore($embedMode)
  const embedAllowed = useStore($embedAllowed)
  const composerPopoutGesturesEnabled = useStore($composerPopoutGesturesEnabled)
  const translucency = useStore($translucency)
  const glassMode = translucency.mode === 'glass' && GLASS_SUPPORTED
  const reactionsEnabled = useStore($reactionsEnabled)
  const backdrop = useStore($backdrop)
  const installs = useStore($marketplaceInstalls)
  const profiles = useStore($profiles)
  const activeProfileKey = normalizeProfileKey(useStore($activeGatewayProfile))
  const a = t.settings.appearance

  // A pointer held on the intensity slider when this overlay closes (Escape
  // mid-drag) never delivers its pointerup here, which would strand the peek
  // counter above zero and ghost the NEXT settings overlay. Unmount drops
  // every outstanding hold.
  useEffect(() => resetTranslucencyPeek, [])

  // Shared by the mode/frost/area pickers: apply the choice, then show it
  // through the overlay it just altered (a pulse, not a hold — see the peek
  // notes on the slider itself).
  const pickTranslucency =
    <T,>(set: (value: T) => void) =>
    (value: T) => {
      triggerHaptic('selection')
      set(value)

      if (translucency.intensity > 0) {
        pulseTranslucencyPeek()
      }
    }

  const [query, setQuery] = useState('')

  useDeepLinkHighlight({
    elementId: appearanceSettingElementId,
    param: 'setting',
    ready: id => APPEARANCE_SEARCH_TARGETS.has(id)
  })

  // One box does double duty: filter installed themes live (below), and run a
  // name search against the VS Code Marketplace (the Cmd-K "Install theme…"
  // backend) for anything not already installed.
  const needle = normalize(query)

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

  const sessionDensityOptions = [
    { id: 'compact', label: a.sessionDensityCompact },
    { id: 'comfortable', label: a.sessionDensityComfortable },
    { id: 'detailed', label: a.sessionDensityDetailed }
  ] as const satisfies readonly { id: SessionListDensity; label: string }[]

  const embedOptions = [
    { id: 'ask', label: a.embedsAsk },
    { id: 'always', label: a.embedsAlways },
    { id: 'off', label: a.embedsOff }
  ] as const satisfies readonly { id: EmbedMode; label: string }[]

  const uiScaleOptions = UI_SCALE_PRESETS.map(preset => ({ id: preset, label: `${preset}%` }))

  const matchedScalePreset = matchUiScalePreset(zoomPercent)

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
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.language)}
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
                {showProfileNote && (
                  <p className="mt-3 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                    {a.themeProfileNote(activeProfileName)}
                  </p>
                )}
              </>
            }
            description={a.themeDesc}
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.theme)}
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
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setZoomPercent(Number(id))
                }}
                options={uiScaleOptions}
                value={matchedScalePreset ?? ('' as UiScalePreset)}
              />
            }
            description={a.uiScaleDesc(zoomPercent)}
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.uiScale)}
            title={a.uiScaleTitle}
          />

          <TerminalFontSetting />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setSessionListDensity(id)
                }}
                options={sessionDensityOptions}
                value={sessionListDensity}
              />
            }
            description={a.sessionDensityDesc}
            title={a.sessionDensityTitle}
          />

          {/* Linux has neither half of this setting (see TRANSLUCENCY_SUPPORTED),
              so the row is absent there rather than offering a dead lever. */}
          {TRANSLUCENCY_SUPPORTED && (
            <ListRow
              action={
                <div
                  className="flex items-center gap-3"
                  // Arms the peek for the overlay this row lives in — the
                  // ghosting rules in styles.css scope to it, so no other
                  // overlay pays for an opacity transition it never uses.
                  data-translucency-peek-scope=""
                >
                  {GLASS_SUPPORTED && (
                    <SegmentedControl
                      onChange={pickTranslucency(setTranslucencyMode)}
                      options={[
                        { id: 'clear' as const, label: a.translucencyModeClear },
                        { id: 'glass' as const, label: a.translucencyModeGlass }
                      ]}
                      value={translucency.mode}
                    />
                  )}
                  {/* Clear has one lever and it belongs beside the mode. Glass
                      has four controls, so they move into the labelled panel
                      below rather than crowding this line with an unlabelled
                      slider that means something different. */}
                  {!glassMode && (
                    <TranslucencySlider
                      label={a.translucencyTitle}
                      onChange={setTranslucency}
                      value={translucency.intensity}
                    />
                  )}
                </div>
              }
              below={
                glassMode ? (
                  <div className="mt-3 flex flex-col gap-2.5" data-translucency-peek-scope="">
                    <GlassRow label={a.translucencyTintTitle}>
                      <TranslucencySlider
                        label={a.translucencyTintTitle}
                        onChange={setTranslucency}
                        value={translucency.intensity}
                      />
                    </GlassRow>
                    <GlassRow label={a.translucencyFadeTitle}>
                      <TranslucencySlider
                        label={a.translucencyFadeTitle}
                        onChange={setTranslucencyFade}
                        value={translucency.fade}
                      />
                    </GlassRow>
                    <GlassRow label={a.translucencyFrostTitle}>
                      <SegmentedControl
                        onChange={pickTranslucency(setTranslucencyMaterial)}
                        // Windows renders four rungs as three backdrops, so it
                        // is offered three; a frost saved on a Mac highlights
                        // the rung that renders the same backdrop here.
                        options={glassMaterialsFor(GLASS_IS_WINDOWS).map(material => ({
                          id: material,
                          label: a.translucencyFrost[material]
                        }))}
                        value={glassMaterialForPicker(translucency.material, GLASS_IS_WINDOWS)}
                      />
                    </GlassRow>
                    <GlassRow label={a.translucencyScopeTitle}>
                      <SegmentedControl
                        onChange={pickTranslucency(setTranslucencyScope)}
                        options={GLASS_SCOPES.map(scope => ({
                          id: scope,
                          label: a.translucencyScope[scope]
                        }))}
                        value={translucency.scope}
                      />
                    </GlassRow>
                  </div>
                ) : undefined
              }
              description={glassMode ? a.translucencyGlassDesc : a.translucencyDesc}
              id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.translucency)}
              title={a.translucencyTitle}
            />
          )}

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setBackdrop(id === 'on')
                }}
                options={[
                  { id: 'off', label: t.common.off },
                  { id: 'on', label: t.common.on }
                ]}
                value={backdrop ? 'on' : 'off'}
              />
            }
            description={a.backdropDesc}
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.backdrop)}
            title={a.backdropTitle}
          />

          <ToggleRow
            checked={composerPopoutGesturesEnabled}
            description={a.composerPopoutDesc}
            label={a.composerPopoutTitle}
            onChange={setComposerPopoutGesturesEnabled}
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setReactionsEnabled(id === 'on')
                }}
                options={[
                  { id: 'off', label: t.common.off },
                  { id: 'on', label: t.common.on }
                ]}
                value={reactionsEnabled ? 'on' : 'off'}
              />
            }
            description={a.reactionsDesc}
            title={a.reactionsTitle}
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
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.toolView)}
            title={a.toolViewTitle}
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setReasoningCollapsedByDefault(id === 'on')
                }}
                options={[
                  { id: 'off', label: t.common.off },
                  { id: 'on', label: t.common.on }
                ]}
                value={reasoningCollapsedByDefault ? 'on' : 'off'}
              />
            }
            description={a.reasoningCollapsedDesc}
            title={a.reasoningCollapsedTitle}
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
            id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.embeds)}
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
