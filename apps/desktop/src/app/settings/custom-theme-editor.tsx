import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { normalizeHex } from '@/themes/color'
import { getBaseColors } from '@/themes/context'
import {
  $themePreview,
  buildCustomTheme,
  createCustomThemeDefinition,
  type CustomThemeDefinition,
  type CustomThemePaletteSeed,
  getCustomThemeDefinition,
  resetCustomThemePalettes,
  saveCustomTheme,
  uniqueCustomThemeName
} from '@/themes/custom-themes'
import { nousTheme } from '@/themes/presets'
import { resolveTheme } from '@/themes/user-themes'

interface CustomThemeEditorProps {
  mode: 'light' | 'dark'
  onOpenChange: (open: boolean) => void
  onSaved: (name: string) => void
  open: boolean
  sourceThemeName: string
}

const cloneDefinition = (definition: CustomThemeDefinition): CustomThemeDefinition => ({
  ...definition,
  light: { ...definition.light },
  dark: { ...definition.dark }
})

function initialDefinition(sourceThemeName: string): {
  definition: CustomThemeDefinition
  editing: boolean
} {
  const saved = getCustomThemeDefinition(sourceThemeName)

  if (saved) {
    return { definition: cloneDefinition(saved), editing: true }
  }

  const source = resolveTheme(sourceThemeName) ?? nousTheme

  return {
    definition: createCustomThemeDefinition({
      source,
      lightColors: getBaseColors(source.name, 'light'),
      darkColors: getBaseColors(source.name, 'dark')
    }),
    editing: false
  }
}

function ColorField({
  label,
  onChange,
  value
}: {
  label: string
  onChange: (value: string) => void
  value: string
}) {
  const valid = normalizeHex(value)
  const swatch = valid ?? '#000000'

  return (
    <label className="grid gap-1">
      <span className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
        {label}
      </span>
      <span
        className={cn(
          'flex items-center gap-2 rounded-lg border bg-(--ui-bg-quinary) px-2 py-1.5',
          valid ? 'border-(--ui-stroke-tertiary)' : 'border-(--ui-red)'
        )}
      >
        <input
          aria-label={`${label} color`}
          className="size-6 shrink-0 cursor-pointer rounded border-0 bg-transparent p-0"
          onChange={event => onChange(event.target.value)}
          type="color"
          value={swatch}
        />
        <input
          aria-invalid={!valid}
          className="min-w-0 flex-1 bg-transparent font-mono text-[length:var(--conversation-caption-font-size)] uppercase outline-none"
          onBlur={() => valid && onChange(valid)}
          onChange={event => onChange(event.target.value)}
          spellCheck={false}
          value={value}
        />
      </span>
    </label>
  )
}

export function CustomThemeEditor({
  mode,
  onOpenChange,
  onSaved,
  open,
  sourceThemeName
}: CustomThemeEditorProps) {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const [draft, setDraft] = useState<CustomThemeDefinition | null>(null)
  const [editing, setEditing] = useState(false)
  const [editingMode, setEditingMode] = useState<'light' | 'dark'>(mode)

  useEffect(() => {
    if (!open) {
      $themePreview.set(null)

      return
    }

    const next = initialDefinition(sourceThemeName)
    setDraft(next.definition)
    setEditing(next.editing)
    setEditingMode(mode)

    return () => $themePreview.set(null)
  }, [mode, open, sourceThemeName])

  const previewTheme = useMemo(() => {
    if (!draft) {
      return null
    }

    try {
      return buildCustomTheme(draft)
    } catch {
      return null
    }
  }, [draft])

  useEffect(() => {
    $themePreview.set(open && previewTheme ? { mode: editingMode, theme: previewTheme } : null)
  }, [editingMode, open, previewTheme])

  const palette = draft?.[editingMode]
  const colors = previewTheme ? (editingMode === 'dark' ? previewTheme.darkColors : previewTheme.colors) : null

  const invalidColor = draft
    ? [draft.light, draft.dark].some(
        seed => !normalizeHex(seed.accent) || !normalizeHex(seed.background) || !normalizeHex(seed.foreground)
      )
    : true

  const invalidName = !draft?.label.trim()
  const invalidFont = !draft?.fontSans.trim() || !draft?.fontMono.trim()
  const canSave = Boolean(draft && !invalidColor && !invalidName && !invalidFont)

  const updateDraft = (patch: Partial<CustomThemeDefinition>) => {
    setDraft(current => (current ? { ...current, ...patch } : current))
  }

  const updatePalette = (patch: Partial<CustomThemePaletteSeed>) => {
    setDraft(current =>
      current
        ? {
            ...current,
            [editingMode]: { ...current[editingMode], ...patch }
          }
        : current
    )
  }

  const close = () => {
    $themePreview.set(null)
    onOpenChange(false)
  }

  const reset = () => {
    if (!draft) {
      return
    }

    triggerHaptic('selection')
    setDraft(current => (current ? resetCustomThemePalettes(current) : current))
    setEditingMode(mode)
  }

  const save = () => {
    if (!draft || !canSave) {
      return
    }

    const finalDraft = editing ? draft : { ...draft, name: uniqueCustomThemeName(draft.label) }
    const theme = saveCustomTheme(finalDraft)

    triggerHaptic('crisp')
    $themePreview.set(null)
    onSaved(theme.name)
    onOpenChange(false)
  }

  const modeOptions = (['light', 'dark'] as const).map(id => ({
    id,
    label: t.settings.modeOptions[id].label
  }))

  return (
    <Dialog
      onOpenChange={next => {
        if (!next) {
          close()
        }
      }}
      open={open}
    >
      <DialogContent className="max-w-2xl gap-4">
        <DialogHeader>
          <DialogTitle icon={Palette}>{editing ? copy.editCustomTheme : copy.createCustomTheme}</DialogTitle>
          <DialogDescription>{copy.customThemeDesc}</DialogDescription>
        </DialogHeader>

        {draft && palette ? (
          <>
            <div className="flex items-center justify-between gap-3">
              <label className="min-w-0 flex-1">
                <span className="mb-1 block text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                  {copy.customThemeName}
                </span>
                <input
                  aria-invalid={invalidName}
                  className="w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 outline-none focus:border-(--ui-stroke-secondary)"
                  onChange={event => updateDraft({ label: event.target.value })}
                  value={draft.label}
                />
              </label>
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setEditingMode(id)
                }}
                options={modeOptions}
                value={editingMode}
              />
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <ColorField
                label={copy.customAccent}
                onChange={accent => updatePalette({ accent })}
                value={palette.accent}
              />
              <ColorField
                label={copy.customBackground}
                onChange={background => updatePalette({ background })}
                value={palette.background}
              />
              <ColorField
                label={copy.customForeground}
                onChange={foreground => updatePalette({ foreground })}
                value={palette.foreground}
              />
            </div>

            {invalidColor && (
              <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-red)" role="alert">
                {copy.customInvalidColor}
              </p>
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="grid gap-1">
                <span className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                  {copy.customUiFont}
                </span>
                <input
                  aria-invalid={!draft.fontSans.trim()}
                  className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 outline-none focus:border-(--ui-stroke-secondary)"
                  onChange={event => updateDraft({ fontSans: event.target.value })}
                  spellCheck={false}
                  value={draft.fontSans}
                />
              </label>
              <label className="grid gap-1">
                <span className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                  {copy.customCodeFont}
                </span>
                <input
                  aria-invalid={!draft.fontMono.trim()}
                  className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 outline-none focus:border-(--ui-stroke-secondary)"
                  onChange={event => updateDraft({ fontMono: event.target.value })}
                  spellCheck={false}
                  value={draft.fontMono}
                />
              </label>
            </div>

            <div className="grid gap-3 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-[length:var(--conversation-caption-font-size)] font-medium">
                  {copy.customSidebar}
                </span>
                <Switch
                  checked={draft.translucentSidebar}
                  onCheckedChange={checked => updateDraft({ translucentSidebar: checked })}
                />
              </div>
              <div className="flex items-center gap-3">
                <label
                  className="min-w-0 flex-1 text-[length:var(--conversation-caption-font-size)] font-medium"
                  htmlFor="custom-theme-contrast"
                >
                  {copy.customContrast}
                </label>
                <input
                  className="h-1 w-48 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
                  id="custom-theme-contrast"
                  max={100}
                  min={0}
                  onChange={event => updatePalette({ contrast: Number(event.target.value) })}
                  style={{ accentColor: 'var(--dt-primary)' }}
                  type="range"
                  value={palette.contrast}
                />
                <span className="w-8 text-right tabular-nums text-(--ui-text-tertiary)">{palette.contrast}</span>
              </div>
            </div>

            {colors && (
              <div
                className="overflow-hidden rounded-xl border"
                style={{ backgroundColor: colors.background, borderColor: colors.border }}
              >
                <div className="flex min-h-24">
                  <div
                    className="w-24 border-r p-3"
                    style={{
                      background: colors.sidebarBackground,
                      borderColor: colors.sidebarBorder,
                      color: colors.foreground
                    }}
                  >
                    <div className="h-2 w-12 rounded-full" style={{ background: colors.primary }} />
                  </div>
                  <div className="flex flex-1 flex-col gap-2 p-4" style={{ color: colors.foreground }}>
                    <div className="h-2.5 w-28 rounded-full" style={{ background: colors.foreground }} />
                    <div className="h-2 w-40 rounded-full" style={{ background: colors.mutedForeground }} />
                    <div
                      className="ml-auto mt-auto h-7 w-32 rounded-lg border"
                      style={{ background: colors.userBubble, borderColor: colors.userBubbleBorder }}
                    />
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}

        <DialogFooter>
          <Button onClick={reset} type="button" variant="text">
            {copy.customReset}
          </Button>
          <Button onClick={close} type="button" variant="outline">
            {t.common.cancel}
          </Button>
          <Button disabled={!canSave} onClick={save} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
