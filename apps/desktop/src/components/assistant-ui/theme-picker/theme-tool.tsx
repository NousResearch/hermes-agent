'use client'

import type { ToolCallMessagePart, ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { LAYOUTS_AREA } from '@/components/pane-shell/tree/presets'
import { Codicon } from '@/components/ui/codicon'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { connectorText, recordOf } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'
import type { AppliedAppearance } from '@/store/appearance-apply'
import { $themePicks, cancelThemePick, pickTheme } from '@/store/theme-pick'
import { useTheme } from '@/themes'
import { accentName, resolveAccentPick } from '@/themes/accents'
import type { ThemeMode } from '@/themes/context'

import { CARD_LABEL_CLASS } from '../setup-beat/card-type'
import { ToolFallback } from '../tool/fallback'
import { TypedAnswer } from '../typed-answer'

import { AccentPickerCard } from './accent-picker-card'
import { setThemeCall } from './args'
import { LayoutPickerCard } from './layout-picker-card'

const SHELL_CLASS = cn(
  WIDGET_SHELL_CLASS,
  'my-1.5 text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)'
)

const MODES: readonly ThemeMode[] = ['light', 'dark', 'system']

/** The applied/picked values of a set_theme result, or null when the result names none. */
function readValues(value: ToolCallMessagePart['result']): AppliedAppearance | null {
  const row = recordOf(value)
  const values: AppliedAppearance = {}
  const accent = row.accent === null ? null : connectorText(row.accent)
  const mode = MODES.find(candidate => candidate === row.mode)
  const layout = connectorText(row.layout)

  if (accent !== undefined) {
    values.accent = accent
  }

  if (mode) {
    values.mode = mode
  }

  if (layout) {
    values.layout = layout
  }

  return Object.keys(values).length > 0 ? values : null
}

/** What a settled set_theme call changed, on one line; `null` = the card closed without a pick. */
export function ThemeReceipt({ values }: { values: AppliedAppearance | null }) {
  const { t } = useI18n()
  const copy = t.assistant.themePicker
  const { renderedMode } = useTheme()
  const presets = useContributions(LAYOUTS_AREA)

  if (!values) {
    return (
      <div className={SHELL_CLASS} data-theme-settled="">
        <p className="italic text-(--ui-text-tertiary)">{copy.noPick}</p>
      </div>
    )
  }

  const dot = values.accent ? resolveAccentPick(values.accent, renderedMode === 'dark')?.hex : null

  return (
    <div className={cn(SHELL_CLASS, 'flex flex-wrap items-center gap-x-4 gap-y-1')} data-theme-settled="">
      {values.accent !== undefined && (
        <span className="flex items-center gap-1.5">
          <span className="text-(--ui-text-tertiary)">{copy.accent}</span>
          {dot && <span aria-hidden className="size-3 rounded-full" style={{ background: dot }} />}
          <span>{values.accent === null ? copy.themeAccent : accentName(values.accent)}</span>
        </span>
      )}
      {values.mode && (
        <span className="flex items-center gap-1.5">
          <span className="text-(--ui-text-tertiary)">{copy.mode}</span>
          <span>{t.settings.modeOptions[values.mode].label}</span>
        </span>
      )}
      {values.layout && (
        <span className="flex items-center gap-1.5">
          <span className="text-(--ui-text-tertiary)">{copy.layout}</span>
          <span>{presets.find(preset => preset.id === values.layout)?.title ?? values.layout}</span>
        </span>
      )}
    </div>
  )
}

/** The set_theme tool row: the picker card while its `theme.pick` waits, then a one-line receipt. */
export function ThemeTool(props: ToolCallMessagePartProps) {
  const copy = useI18n().t.assistant.themePicker
  const sessionId = useStore(useSessionView().$runtimeId)
  const picks = useStore($themePicks)
  const [picked, setPicked] = useState<AppliedAppearance | null>(null)
  const picker = setThemeCall(props.toolName, props.args)?.picker ?? null

  if (props.result !== undefined) {
    const result = recordOf(props.result)
    const said = connectorText(result.said)?.trim()

    if (said) {
      return <TypedAnswer text={said} />
    }

    const values = readValues(result.applied ?? result.picked)

    if (values || (picker && result.success !== false)) {
      return <ThemeReceipt values={values} />
    }

    return <ToolFallback {...props} />
  }

  // Answered: the card is gone from the store a beat before tool.complete brings the result.
  if (picked) {
    return <ThemeReceipt values={picked} />
  }

  const request = sessionId ? picks[sessionId] : undefined

  if (!sessionId || !picker || request?.picker !== picker) {
    return <ToolFallback {...props} />
  }

  const answer = (values: AppliedAppearance, apply: () => boolean): boolean => {
    if (!apply()) {
      return false
    }

    setPicked(values)

    return true
  }

  return (
    <div className={cn(SHELL_CLASS, 'grid gap-2')} data-theme-picker={picker}>
      <div className="flex items-center justify-between gap-2">
        <span className={CARD_LABEL_CLASS}>{copy[picker]}</span>
        <button
          aria-label={copy.noPick}
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
          onClick={() => cancelThemePick(sessionId)}
          type="button"
        >
          <Codicon name="close" size="0.8rem" />
        </button>
      </div>
      {picker === 'accent' ? (
        <AccentPickerCard
          onContinue={accent => answer({ accent }, () => pickTheme(sessionId, { accent: accent ?? 'theme' }))}
        />
      ) : (
        <LayoutPickerCard onContinue={layout => answer({ layout }, () => pickTheme(sessionId, { layout }))} />
      )}
    </div>
  )
}
