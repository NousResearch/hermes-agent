'use client'

import type { SetupChooseKind } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { type CSSProperties, type FC, type ReactNode, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Chip } from '@/components/onboarding-chat/chip'
import { AccentSwatch, LayoutPreviewCard, LAYOUTS, NOUS_ACCENT } from '@/components/onboarding-chat/options'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Kbd } from '@/components/ui/kbd'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { connectorIconUrl } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'
import type { ClarifyQuestion } from '@/store/clarify'
import { pluginNeedsApp, useOnboardingPluginList } from '@/store/onboarding-plugins'
import { useTheme } from '@/themes'
import { getBaseColors } from '@/themes/context'

import { ChoiceLabel, letterFor } from './core/choice-row'
import type { SetupRow } from './setup-rows'

export interface SetupPickerProps {
  cursor: null | number
  onPick: (index: number) => void
  onStage: (id: string) => void
  picked: string[]
  rows: SetupRow[]
}

const SEARCH_THRESHOLD = 12

const PILL_MAX_OPTIONS = 6

const PILL_MAX_LABEL = 40

export const PICKER_COLUMNS: Record<Exclude<SetupChooseKind, 'question'>, (rows: SetupRow[]) => number> = {
  accent: rows => rows.length,
  connectors: () => 3,
  layout: () => 2,
  plugins: () => 3,
  theme: rows => rows.length
}

export const pillQuestion = (rows: SetupRow[]): boolean =>
  rows.length > 0 &&
  rows.length <= PILL_MAX_OPTIONS &&
  rows.every(row => !row.detail && row.label.length <= PILL_MAX_LABEL)

export function pickerShortcutCount(rows: SetupRow[]): number {
  return rows.length > SEARCH_THRESHOLD ? 0 : rows.length
}

function PickerItem({
  active,
  children,
  className,
  index
}: {
  active: boolean
  children: ReactNode
  className?: string
  index: null | number
}) {
  return (
    <div
      aria-keyshortcuts={index === null ? undefined : `${letterFor(index)} ${index + 1}`}
      className={cn('grid min-w-0', active && 'ring-2 ring-primary/40', className)}
      data-highlighted={active || undefined}
      onMouseDown={event => event.preventDefault()}
    >
      {children}
    </div>
  )
}

function ThemePicker({ cursor, onPick, picked, rows }: SetupPickerProps) {
  const { themeName } = useTheme()
  const light = getBaseColors(themeName, 'light')
  const dark = getBaseColors(themeName, 'dark')

  const palettes: Record<string, CSSProperties> = {
    dark: {
      background: dark.background,
      borderColor: dark.border,
      '--color-foreground': dark.foreground
    } as CSSProperties,
    light: {
      background: light.background,
      borderColor: light.border,
      '--color-foreground': light.foreground
    } as CSSProperties,
    system: {
      background: `linear-gradient(135deg, ${light.background} 50%, ${dark.background} 50%)`,
      borderColor: light.border,
      '--color-foreground': '#808080'
    } as CSSProperties
  }

  return (
    <div className="grid grid-cols-3 gap-3 p-1" role="group">
      {rows.map((row, index) => {
        const active = picked.includes(row.id)
        const palette = palettes[row.id]

        return (
          <PickerItem active={cursor === index} className="rounded-[8px]" index={index} key={row.id}>
            <LayoutPreviewCard
              active={active}
              name={row.label}
              onSelect={() => onPick(index)}
              previewStyle={active && palette ? { ...palette, borderColor: undefined } : palette}
              tree={LAYOUTS[0].tree}
            />
          </PickerItem>
        )
      })}
    </div>
  )
}

function AccentPicker({ cursor, onPick, onStage, picked, rows }: SetupPickerProps) {
  const { t } = useI18n()
  const current = picked[0] ?? NOUS_ACCENT
  const custom = !rows.some(row => row.id.toLowerCase() === current.toLowerCase())

  return (
    <div className="flex flex-wrap gap-2.5 p-1" role="group">
      {rows.map((row, index) => (
        <PickerItem active={cursor === index} className="rounded-full" index={index} key={row.id}>
          <AccentSwatch active={picked.includes(row.id)} hex={row.id} name={row.label} onPick={() => onPick(index)} />
        </PickerItem>
      ))}
      <AccentSwatch
        active={picked.length > 0 && custom}
        hex={custom ? current : NOUS_ACCENT}
        name={t.assistant.setupChoose.customColor}
        onColorChange={onStage}
      />
    </div>
  )
}

function LayoutPicker({ cursor, onPick, picked, rows }: SetupPickerProps) {
  return (
    <div className="grid grid-cols-2 gap-3 p-1" role="group">
      {rows.map((row, index) => (
        <PickerItem active={cursor === index} className="rounded-[8px]" index={index} key={row.id}>
          <LayoutPreviewCard
            active={picked.includes(row.id)}
            description={row.detail ?? undefined}
            name={row.label}
            onSelect={() => onPick(index)}
            tree={LAYOUTS.find(layout => layout.id === row.id)?.tree ?? 1}
          />
        </PickerItem>
      ))}
    </div>
  )
}

function ChipPicker({
  cursor,
  dim,
  icon,
  onPick,
  picked,
  rows,
  sub
}: SetupPickerProps & {
  dim?: (row: SetupRow) => boolean
  icon: (row: SetupRow) => ReactNode
  sub?: string
}) {
  const { t } = useI18n()
  const [query, setQuery] = useState('')
  const search = query.trim().toLowerCase()
  const shortcuts = pickerShortcutCount(rows)

  return (
    <div className="grid gap-2">
      {rows.length > SEARCH_THRESHOLD ? (
        <SearchField onChange={setQuery} placeholder={t.assistant.setupChoose.findApp} value={query} />
      ) : null}
      <div className="grid max-h-72 grid-cols-3 gap-2 overflow-y-auto p-1" role="group">
        {rows.map((row, index) =>
          search && !row.label.toLowerCase().includes(search) ? null : (
            <PickerItem
              active={cursor === index}
              className="rounded-[6px]"
              index={index < shortcuts ? index : null}
              key={row.id}
            >
              <Chip
                className={cn('w-full', dim?.(row) && 'opacity-60')}
                icon={icon(row)}
                label={row.label}
                on={picked.includes(row.id)}
                onToggle={() => onPick(index)}
                sub={row.detail ?? sub}
              />
            </PickerItem>
          )
        )}
      </div>
      <p className="text-xs text-muted-foreground">{t.assistant.setupChoose.startsLater}</p>
    </div>
  )
}

const connectorIcon = (row: SetupRow) => (
  <ConnectorLogo
    className="size-7 rounded-full text-sm"
    connector={{ iconUrl: connectorIconUrl(row.id), name: row.id, title: row.label }}
  />
)

const pluginIcon = () => (
  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-muted-foreground">
    <Puzzle className="size-4" />
  </span>
)

function ConnectorPicker(props: SetupPickerProps) {
  return <ChipPicker {...props} icon={connectorIcon} />
}

function PluginPicker(props: SetupPickerProps) {
  const { t } = useI18n()
  const plugins = useOnboardingPluginList(useStore(useSessionView().$storedId))

  const needsApp = (row: SetupRow) => Boolean(plugins?.some(plugin => plugin.name === row.id && pluginNeedsApp(plugin)))

  return (
    <ChipPicker
      {...props}
      dim={needsApp}
      icon={pluginIcon}
      sub={t.assistant.setupChoose.plugin}
    />
  )
}

export const SETUP_PICKERS: Record<Exclude<SetupChooseKind, 'question'>, FC<SetupPickerProps>> = {
  accent: AccentPicker,
  connectors: ConnectorPicker,
  layout: LayoutPicker,
  plugins: PluginPicker,
  theme: ThemePicker
}

const PILL_CLASS =
  'flex max-w-full shrink-0 items-center gap-1.5 rounded-full border py-1 pr-3 pl-1.5 text-left text-[12px] whitespace-normal wrap-anywhere transition-colors disabled:cursor-not-allowed disabled:opacity-50'

const PILL_CURSOR_CLASS = 'ring-2 ring-ring/60 ring-offset-2 ring-offset-(--dt-background)'

export function QuestionPills({
  cursor,
  disabled,
  onActivate,
  onDraft,
  onOtherFocus,
  onPick,
  question,
  staged
}: {
  cursor: null | number
  disabled: boolean
  onActivate: () => void
  onDraft: (value: string) => void
  onOtherFocus: () => void
  onPick: (index: number) => void
  question: ClarifyQuestion
  staged: { choices: string[]; draft: string }
}) {
  const { t } = useI18n()
  const choices = question.choices ?? []
  const otherActive = cursor === choices.length
  const drafted = Boolean(staged.draft.trim())

  return (
    <div
      className="grid gap-2"
      data-clarify-batch-question={question.qid}
      onFocus={onActivate}
      onPointerDown={onActivate}
    >
      <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question.question}</span>
      <div className="flex min-w-0 flex-wrap gap-2 p-1" role="group">
        {choices.map((choice, index) => {
          const selected = staged.choices.includes(choice)

          return (
            <button
              aria-current={cursor === index || undefined}
              aria-keyshortcuts={cursor === null ? undefined : `${letterFor(index)} ${index + 1}`}
              aria-pressed={selected}
              className={cn(
                PILL_CLASS,
                selected
                  ? 'border-primary bg-primary text-primary-foreground'
                  : 'border-border bg-card hover:border-primary/50 hover:bg-primary/10',
                cursor === index && PILL_CURSOR_CLASS
              )}
              data-choice
              data-highlighted={cursor === index || undefined}
              disabled={disabled}
              key={`${index}-${choice}`}
              onClick={() => onPick(index)}
              type="button"
            >
              <Kbd size="sm" variant={selected ? 'inverted' : 'default'}>
                {letterFor(index)}
              </Kbd>
              <span>
                <ChoiceLabel choice={choice} />
              </span>
            </button>
          )
        })}
        <label
          className={cn(
            PILL_CLASS,
            'cursor-text',
            drafted ? 'border-primary bg-primary/10' : 'border-border bg-card',
            otherActive && PILL_CURSOR_CLASS,
            disabled && 'cursor-not-allowed opacity-50'
          )}
          data-highlighted={otherActive || undefined}
        >
          <Kbd size="sm">{letterFor(choices.length)}</Kbd>
          <textarea
            aria-current={otherActive || undefined}
            aria-keyshortcuts={cursor === null ? undefined : `${letterFor(choices.length)} ${choices.length + 1}`}
            className="field-sizing-content max-h-40 min-w-8 resize-none bg-transparent leading-5 outline-none placeholder:text-muted-foreground focus:min-w-48"
            disabled={disabled}
            onChange={event => onDraft(event.target.value)}
            onFocus={onOtherFocus}
            placeholder={t.assistant.clarify.other}
            rows={1}
            value={staged.draft}
          />
        </label>
      </div>
    </div>
  )
}
