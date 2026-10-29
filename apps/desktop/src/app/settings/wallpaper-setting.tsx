import { useStore } from '@nanostores/react'
import { useEffect, useEffectEvent, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { FileImage, Loader2, Trash2 } from '@/lib/icons'
import {
  DEFAULT_MANUAL_WALLPAPER_PALETTE,
  WALLPAPER_MASK_SHAPES,
  WALLPAPER_MODES,
  WALLPAPER_PALETTE_MODES,
  wallpaperBackgroundProperties,
  wallpaperMaskImage,
  type WallpaperMaskShape,
  type WallpaperMode,
  type WallpaperPaletteMode
} from '@/lib/wallpaper'
import { $wallpaper, setWallpaperPreferences } from '@/store/wallpaper'
import {
  ensureWallpaperLoaded,
  removeWallpaper,
  resetWallpaperPreferences,
  selectWallpaper,
  setWallpaperAdaptiveTheme,
  setWallpaperPaletteMode
} from '@/store/wallpaper-actions'
import { previewWallpaperThemePalette, restoreWallpaperThemePreview } from '@/themes/context'

import { ListRow } from './primitives'

export function WallpaperSlider({
  disabled,
  label,
  max,
  min,
  onChange,
  step = 1,
  value,
  valueLabel
}: {
  disabled: boolean
  label: string
  max: number
  min: number
  onChange: (value: number) => void
  step?: number
  value: number
  valueLabel: string
}) {
  const changeRef = useRef(onChange)
  const pendingValueRef = useRef(value)
  const frameRef = useRef<number | null>(null)

  changeRef.current = onChange

  if (frameRef.current === null) {
    pendingValueRef.current = value
  }

  const flushPendingValue = () => {
    if (frameRef.current === null) {
      return
    }

    window.cancelAnimationFrame(frameRef.current)
    frameRef.current = null
    changeRef.current(pendingValueRef.current)
  }

  const scheduleValue = (nextValue: number) => {
    pendingValueRef.current = nextValue

    if (frameRef.current !== null) {
      return
    }

    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null
      changeRef.current(pendingValueRef.current)
    })
  }

  useEffect(
    () => () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current)
      }
    },
    []
  )

  return (
    <label className={disabled ? 'opacity-50' : undefined}>
      <span className="mb-1.5 flex items-center justify-between gap-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
        <span>{label}</span>
        <span className="tabular-nums text-(--ui-text-tertiary)">{valueLabel}</span>
      </span>
      <input
        aria-label={label}
        className="h-1 w-full cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary) disabled:cursor-default"
        disabled={disabled}
        max={max}
        min={min}
        onBlur={flushPendingValue}
        onChange={event => scheduleValue(Number(event.target.value))}
        onKeyUp={flushPendingValue}
        onPointerUp={flushPendingValue}
        step={step}
        style={{ accentColor: 'var(--dt-primary)' }}
        type="range"
        value={value}
      />
    </label>
  )
}

export function WallpaperColorControl({
  disabled,
  label,
  onChange,
  onPreview,
  onPreviewEnd,
  themeColor = 'var(--ui-chat-surface-background)',
  themeLabel,
  value
}: {
  disabled: boolean
  label: string
  onChange: (value: string) => void
  onPreview?: (value: string) => void
  onPreviewEnd?: () => void
  themeColor?: string
  themeLabel?: string
  value: string
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const commitColor = useEffectEvent((color: string) => onChange(color))
  const previewColor = useEffectEvent((color: string) => onPreview?.(color))
  const finishPreview = useEffectEvent(() => onPreviewEnd?.())

  useEffect(() => {
    const input = inputRef.current

    if (input) {
      input.value = value || '#ffffff'
    }
  }, [value])

  useEffect(() => {
    const input = inputRef.current

    if (!input) {
      return
    }

    let previewFrame: number | null = null
    let restoreFrame: number | null = null
    let latestColor = input.value
    let accepted = true

    // Keep drag feedback off React and the persisted wallpaper store. One
    // imperative CSS update per animation frame is enough for live feedback
    // without multiplying theme renders by the color plane's event rate.
    const preview = () => {
      latestColor = input.value

      if (previewFrame === null) {
        previewFrame = window.requestAnimationFrame(() => {
          previewFrame = null
          previewColor(latestColor)
        })
      }
    }

    const restore = () => {
      if (restoreFrame === null) {
        restoreFrame = window.requestAnimationFrame(() => {
          restoreFrame = null

          if (!accepted) {
            finishPreview()
          }
        })
      }
    }

    const watchPickerClose = () => {
      accepted = false
      window.addEventListener('focus', restore, { once: true })
    }

    // The native `change` event is emitted once when Chromium accepts the
    // picker value. Persist only here; React's onChange maps to `input` and is
    // intentionally not used.
    const commit = () => {
      latestColor = input.value
      accepted = true

      if (previewFrame !== null) {
        window.cancelAnimationFrame(previewFrame)
        previewFrame = null
      }

      previewColor(latestColor)
      commitColor(latestColor)
    }

    input.addEventListener('click', watchPickerClose)
    input.addEventListener('input', preview)
    input.addEventListener('change', commit)

    return () => {
      input.removeEventListener('click', watchPickerClose)
      input.removeEventListener('input', preview)
      input.removeEventListener('change', commit)
      window.removeEventListener('focus', restore)

      if (previewFrame !== null) {
        window.cancelAnimationFrame(previewFrame)
      }

      if (restoreFrame !== null) {
        window.cancelAnimationFrame(restoreFrame)
      }

      finishPreview()
    }
  }, [])

  return (
    <div className={disabled ? 'opacity-50' : undefined}>
      <div className="mb-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
        {label}
      </div>
      <div className="flex min-h-8 items-center gap-2">
        <label className="relative flex h-8 min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) px-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
          <span
            aria-hidden
            className="size-4 shrink-0 rounded-sm border border-(--ui-stroke-secondary)"
            style={{ backgroundColor: value || themeColor }}
          />
          <span className="truncate font-mono">{value ? value.toUpperCase() : themeLabel}</span>
          <input
            aria-label={label}
            className="absolute inset-0 cursor-pointer opacity-0 disabled:cursor-default"
            defaultValue={value || '#ffffff'}
            disabled={disabled}
            ref={inputRef}
            type="color"
          />
        </label>
        {value && themeLabel && (
          <Button disabled={disabled} onClick={() => onChange('')} size="inline" type="button" variant="text">
            {themeLabel}
          </Button>
        )}
      </div>
    </div>
  )
}

export function WallpaperSetting({ profileName }: { profileName: string }) {
  const { t } = useI18n()
  const state = useStore($wallpaper)
  const [previewMode, setPreviewMode] = useState<'effect' | 'full'>('effect')
  const copy = t.settings.appearance.wallpaper
  const hasWallpaper = Boolean(state.asset)

  const busy =
    state.status === 'idle' || state.status === 'loading' || state.status === 'removing' || state.status === 'selecting'

  const background = wallpaperBackgroundProperties(state.preferences.mode)

  useEffect(() => {
    void ensureWallpaperLoaded()
  }, [])

  const mask = wallpaperMaskImage(
    state.preferences.overlayShape,
    state.preferences.overlayX,
    state.preferences.overlayWidth,
    state.preferences.overlayHeight,
    state.preferences.overlayFeather
  )

  const modeOptions = WALLPAPER_MODES.map(id => ({ id, label: copy.modes[id] }))
  const maskShapeOptions = WALLPAPER_MASK_SHAPES.map(id => ({ id, label: copy.overlayShapes[id] }))
  const paletteModeOptions = WALLPAPER_PALETTE_MODES.map(id => ({ id, label: copy.paletteModes[id] }))
  const manualPalette = state.preferences.manualPalette ?? state.preferences.palette ?? DEFAULT_MANUAL_WALLPAPER_PALETTE
  const previewShowsEffect = previewMode === 'effect'

  const positionLabel =
    state.preferences.overlayX === 50
      ? copy.center
      : state.preferences.overlayX < 50
        ? `${copy.left} ${50 - state.preferences.overlayX}`
        : `${copy.right} ${state.preferences.overlayX - 50}`

  return (
    <ListRow
      below={
        <div className="mt-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
          {!state.supported ? (
            <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {copy.unsupported}
            </p>
          ) : !hasWallpaper ? (
            <div className="flex min-h-28 flex-col items-center justify-center gap-2 rounded-md border border-dashed border-(--ui-stroke-secondary) px-4 py-5 text-center">
              {busy ? (
                <Loader2 className="size-5 animate-spin text-(--ui-text-tertiary)" />
              ) : (
                <FileImage className="size-5 text-(--ui-text-tertiary)" />
              )}
              <Button
                disabled={busy}
                onClick={() => void selectWallpaper()}
                size="sm"
                type="button"
                variant="secondary"
              >
                {copy.choose}
              </Button>
            </div>
          ) : (
            <>
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                  {copy.preview}
                </span>
                <SegmentedControl<'effect' | 'full'>
                  disabled={busy}
                  onChange={setPreviewMode}
                  options={[
                    { id: 'effect', label: copy.previewEffect },
                    { id: 'full', label: copy.previewFull }
                  ]}
                  value={previewMode}
                />
              </div>

              <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_7rem] lg:items-start">
                <div className="relative aspect-video overflow-hidden rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)">
                  <div
                    className="absolute"
                    style={{
                      backgroundImage: `url(${JSON.stringify(state.asset?.url)})`,
                      backgroundPosition: previewShowsEffect ? background.position : 'center',
                      backgroundRepeat: previewShowsEffect ? background.repeat : 'no-repeat',
                      backgroundSize: previewShowsEffect ? background.size : 'contain',
                      filter:
                        previewShowsEffect && state.preferences.blur > 0
                          ? `blur(${state.preferences.blur}px)`
                          : undefined,
                      inset: previewShowsEffect ? `-${Math.max(2, state.preferences.blur * 2)}px` : 0,
                      opacity: previewShowsEffect ? state.preferences.opacity / 100 : 1
                    }}
                  />
                  {previewShowsEffect && (
                    <>
                      <div
                        className="absolute inset-0"
                        style={{
                          backgroundColor: state.preferences.overlayColor || 'var(--ui-chat-surface-background)',
                          maskImage: mask,
                          opacity: state.preferences.overlay / 100,
                          WebkitMaskImage: mask
                        }}
                      />
                      <div className="absolute inset-y-0 left-0 w-[24%] border-r border-(--ui-stroke-tertiary)">
                        <div className="absolute inset-0 flex flex-col gap-[7%] p-[10%]">
                          <div className="h-2 w-4/5 rounded-full bg-(--ui-text-secondary)/55" />
                          <div className="h-2 w-full rounded-full bg-(--ui-text-secondary)/45" />
                          <div className="h-2 w-3/4 rounded-full bg-(--ui-text-secondary)/45" />
                          <div className="mt-[12%] h-1.5 w-2/3 rounded-full bg-(--ui-text-tertiary)/40" />
                          <div className="h-1.5 w-full rounded-full bg-(--ui-text-tertiary)/35" />
                        </div>
                      </div>
                      <div className="absolute inset-y-0 right-0 left-[24%] flex flex-col p-[4%]">
                        <div className="h-[14%] w-2/3 self-end rounded-md border border-(--ui-stroke-tertiary) bg-(--dt-user-bubble)/80" />
                        <div className="mt-[5%] flex w-3/5 flex-col gap-2">
                          <div className="h-2 w-2/3 rounded-full bg-(--ui-text-primary)/65" />
                          <div className="h-1.5 w-full rounded-full bg-(--ui-text-secondary)/45" />
                          <div className="h-1.5 w-4/5 rounded-full bg-(--ui-text-secondary)/45" />
                        </div>
                        <div className="mt-auto h-[13%] rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background)/80" />
                      </div>
                    </>
                  )}
                </div>

                <div className="flex flex-wrap gap-2 lg:w-28 lg:flex-col">
                  <Button
                    disabled={busy}
                    onClick={() => {
                      triggerHaptic('crisp')
                      void selectWallpaper()
                    }}
                    size="sm"
                    type="button"
                    variant="secondary"
                  >
                    {busy && state.status === 'selecting' && <Loader2 className="animate-spin" />}
                    {copy.replace}
                  </Button>
                  <Button
                    disabled={busy}
                    onClick={() => {
                      triggerHaptic('selection')
                      void removeWallpaper()
                    }}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    <Trash2 />
                    {copy.remove}
                  </Button>
                </div>
              </div>

              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <div>
                  <div className="mb-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                    {copy.visibility}
                  </div>
                  <SegmentedControl
                    disabled={busy}
                    onChange={id => {
                      triggerHaptic('selection')
                      setWallpaperPreferences({ enabled: id === 'on' })
                    }}
                    options={[
                      { id: 'off', label: t.common.off },
                      { id: 'on', label: t.common.on }
                    ]}
                    value={state.preferences.enabled ? 'on' : 'off'}
                  />
                </div>
                <div>
                  <div className="mb-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                    {copy.displayMode}
                  </div>
                  <SegmentedControl<WallpaperMode>
                    className="max-w-full"
                    disabled={busy || !state.preferences.enabled}
                    onChange={mode => {
                      triggerHaptic('selection')
                      setWallpaperPreferences({ mode })
                    }}
                    options={modeOptions}
                    value={state.preferences.mode}
                  />
                </div>

                <div className="sm:col-span-2">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                        {copy.adaptiveTheme}
                      </div>
                      <p className="mt-1 text-[0.6875rem] leading-relaxed text-(--ui-text-tertiary)">
                        {copy.adaptiveThemeDesc}
                      </p>
                    </div>
                    <SegmentedControl
                      disabled={busy || !state.preferences.enabled}
                      onChange={id => {
                        triggerHaptic('selection')
                        void setWallpaperAdaptiveTheme(id === 'on')
                      }}
                      options={[
                        { id: 'off', label: t.common.off },
                        { id: 'on', label: t.common.on }
                      ]}
                      value={state.preferences.adaptiveTheme ? 'on' : 'off'}
                    />
                  </div>

                  {state.preferences.adaptiveTheme && (
                    <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                        {copy.paletteMode}
                      </span>
                      <SegmentedControl<WallpaperPaletteMode>
                        disabled={busy || !state.preferences.enabled}
                        onChange={mode => {
                          triggerHaptic('selection')
                          void setWallpaperPaletteMode(mode)
                        }}
                        options={paletteModeOptions}
                        value={state.preferences.paletteMode}
                      />
                    </div>
                  )}

                  {state.preferences.adaptiveTheme &&
                    state.preferences.paletteMode === 'auto' &&
                    state.paletteStatus === 'loading' && (
                      <div className="mt-2 flex items-center gap-2 text-[0.6875rem] text-(--ui-text-tertiary)">
                        <Loader2 aria-hidden className="size-3.5 animate-spin" />
                        <span>{copy.adaptiveThemeAnalyzing}</span>
                      </div>
                    )}

                  {state.preferences.adaptiveTheme &&
                    state.preferences.paletteMode === 'auto' &&
                    state.paletteStatus === 'error' && (
                      <p className="mt-2 text-[0.6875rem] text-(--ui-red)" role="alert">
                        {copy.adaptiveThemeError}
                      </p>
                    )}

                  {state.preferences.adaptiveTheme &&
                    state.preferences.paletteMode === 'auto' &&
                    state.preferences.palette && (
                      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
                        {(
                          [
                            [copy.paletteDominant, state.preferences.palette.dominant],
                            [copy.paletteAccent, state.preferences.palette.accent]
                          ] as const
                        ).map(([label, color]) => (
                          <span className="flex items-center gap-1.5" key={label}>
                            <span
                              aria-hidden
                              className="size-3 rounded-sm border border-(--ui-stroke-secondary)"
                              style={{ backgroundColor: color }}
                            />
                            <span>{label}</span>
                            <span className="font-mono">{color.toUpperCase()}</span>
                          </span>
                        ))}
                      </div>
                    )}

                  {state.preferences.adaptiveTheme && state.preferences.paletteMode === 'manual' && (
                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <WallpaperColorControl
                        disabled={busy || !state.preferences.enabled}
                        label={copy.paletteDominant}
                        onChange={dominant =>
                          setWallpaperPreferences({ manualPalette: { ...manualPalette, dominant } })
                        }
                        onPreview={dominant => previewWallpaperThemePalette({ ...manualPalette, dominant })}
                        onPreviewEnd={restoreWallpaperThemePreview}
                        value={manualPalette.dominant}
                      />
                      <WallpaperColorControl
                        disabled={busy || !state.preferences.enabled}
                        label={copy.paletteAccent}
                        onChange={accent => setWallpaperPreferences({ manualPalette: { ...manualPalette, accent } })}
                        onPreview={accent => previewWallpaperThemePalette({ ...manualPalette, accent })}
                        onPreviewEnd={restoreWallpaperThemePreview}
                        value={manualPalette.accent}
                      />
                      <p className="text-[0.6875rem] leading-relaxed text-(--ui-text-tertiary) sm:col-span-2">
                        {copy.paletteManualDesc}
                      </p>
                    </div>
                  )}
                </div>

                <WallpaperSlider
                  disabled={busy || !state.preferences.enabled}
                  label={copy.opacity}
                  max={100}
                  min={0}
                  onChange={opacity => setWallpaperPreferences({ opacity })}
                  step={2}
                  value={state.preferences.opacity}
                  valueLabel={`${state.preferences.opacity}%`}
                />
                <WallpaperSlider
                  disabled={busy || !state.preferences.enabled}
                  label={copy.blur}
                  max={24}
                  min={0}
                  onChange={blur => setWallpaperPreferences({ blur })}
                  value={state.preferences.blur}
                  valueLabel={`${state.preferences.blur}px`}
                />
                <div>
                  <div className="mb-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
                    {copy.overlayShape}
                  </div>
                  <SegmentedControl<WallpaperMaskShape>
                    disabled={busy || !state.preferences.enabled}
                    onChange={overlayShape => {
                      triggerHaptic('selection')
                      setWallpaperPreferences({ overlayShape })
                    }}
                    options={maskShapeOptions}
                    value={state.preferences.overlayShape}
                  />
                </div>
                <WallpaperColorControl
                  disabled={busy || !state.preferences.enabled}
                  label={copy.overlayColor}
                  onChange={overlayColor => setWallpaperPreferences({ overlayColor })}
                  themeLabel={copy.overlayColorTheme}
                  value={state.preferences.overlayColor}
                />
                <WallpaperSlider
                  disabled={busy || !state.preferences.enabled}
                  label={copy.overlay}
                  max={100}
                  min={0}
                  onChange={overlay => setWallpaperPreferences({ overlay })}
                  step={2}
                  value={state.preferences.overlay}
                  valueLabel={`${state.preferences.overlay}%`}
                />
                <WallpaperSlider
                  disabled={busy || !state.preferences.enabled}
                  label={copy.overlayFeather}
                  max={100}
                  min={0}
                  onChange={overlayFeather => setWallpaperPreferences({ overlayFeather })}
                  step={2}
                  value={state.preferences.overlayFeather}
                  valueLabel={`${state.preferences.overlayFeather}%`}
                />
                <div>
                  <WallpaperSlider
                    disabled={busy || !state.preferences.enabled}
                    label={copy.overlayPosition}
                    max={100}
                    min={0}
                    onChange={overlayX => setWallpaperPreferences({ overlayX })}
                    step={2}
                    value={state.preferences.overlayX}
                    valueLabel={positionLabel}
                  />
                  <div className="mt-1 flex justify-between text-[0.6875rem] text-(--ui-text-tertiary)">
                    <span>{copy.left}</span>
                    <span>{copy.center}</span>
                    <span>{copy.right}</span>
                  </div>
                </div>
                <WallpaperSlider
                  disabled={busy || !state.preferences.enabled}
                  label={copy.overlayWidth}
                  max={140}
                  min={30}
                  onChange={overlayWidth => setWallpaperPreferences({ overlayWidth })}
                  step={2}
                  value={state.preferences.overlayWidth}
                  valueLabel={`${state.preferences.overlayWidth}%`}
                />
                {state.preferences.overlayShape === 'ellipse' && (
                  <WallpaperSlider
                    disabled={busy || !state.preferences.enabled}
                    label={copy.overlayHeight}
                    max={200}
                    min={40}
                    onChange={overlayHeight => setWallpaperPreferences({ overlayHeight })}
                    step={2}
                    value={state.preferences.overlayHeight}
                    valueLabel={`${state.preferences.overlayHeight}%`}
                  />
                )}
              </div>

              <div className="mt-4 flex items-center justify-between gap-3 border-t border-(--ui-stroke-tertiary) pt-3">
                <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {copy.profileNote(profileName)}
                </span>
                <Button
                  disabled={busy}
                  onClick={() => {
                    triggerHaptic('selection')
                    resetWallpaperPreferences()
                  }}
                  size="inline"
                  type="button"
                  variant="text"
                >
                  {copy.reset}
                </Button>
              </div>
            </>
          )}

          {state.error && (
            <p className="mt-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-red)" role="alert">
              {copy.error}
            </p>
          )}
        </div>
      }
      description={copy.description}
      title={copy.title}
      wide
    />
  )
}
