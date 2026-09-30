'use client'

import type { SetupChooseIntent, SetupChooseKind } from '@hermes/shared'
import { Puzzle } from 'lucide-react'
import { type FC, type ReactNode, useState } from 'react'

import { Chip } from '@/components/onboarding-chat/chip'
import { AccentSwatch, LayoutPreviewCard, LAYOUTS, NOUS_ACCENT } from '@/components/onboarding-chat/options'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { connectorIconUrl } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'

import { ChoiceButton, letterFor } from './core/choice-row'
import type { SetupRow } from './setup-rows'

export interface SetupPickerProps {
  cursor: null | number
  onPick: (index: number) => void
  onStage: (id: string) => void
  picked: string[]
  rows: SetupRow[]
}

const SEARCH_THRESHOLD = 12

const INTENTS: readonly SetupChooseIntent[] = ['now', 'later', 'save']

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
      className={cn('min-w-0', active && 'ring-2 ring-primary/40', className)}
      data-highlighted={active || undefined}
      onMouseDown={event => event.preventDefault()}
    >
      {children}
    </div>
  )
}

function ThemePicker({ cursor, onPick, picked, rows }: SetupPickerProps) {
  return (
    <div className="grid gap-px" role="group">
      {rows.map((row, index) => (
        <ChoiceButton
          active={cursor === index}
          char={letterFor(index)}
          choice={row.label}
          key={row.id}
          keyShortcuts={`${letterFor(index)} ${index + 1}`}
          onClick={() => onPick(index)}
          selected={picked.includes(row.id)}
        />
      ))}
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
  icon,
  onPick,
  picked,
  rows,
  sub
}: SetupPickerProps & { icon: (row: SetupRow) => ReactNode; sub?: string }) {
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
                className="w-full"
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

  return <ChipPicker {...props} icon={pluginIcon} sub={t.assistant.setupChoose.plugin} />
}

export const SETUP_PICKERS: Record<Exclude<SetupChooseKind, 'question'>, FC<SetupPickerProps>> = {
  accent: AccentPicker,
  connectors: ConnectorPicker,
  layout: LayoutPicker,
  plugins: PluginPicker,
  theme: ThemePicker
}

export function SetupIntentRows({
  intents,
  onIntent,
  rows
}: {
  intents: Record<string, SetupChooseIntent>
  onIntent: (id: string, intent: SetupChooseIntent) => void
  rows: SetupRow[]
}) {
  const { t } = useI18n()
  const options = INTENTS.map(id => ({ id, label: t.assistant.setupChoose.intent[id] }))

  if (rows.length === 0) {
    return null
  }

  return (
    <div className="grid gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2.5">
      {rows.map(row => (
        <div className="flex items-center justify-between gap-2" key={row.id}>
          <span className="min-w-0 truncate text-(--ui-text-secondary)">{row.label}</span>
          <SegmentedControl
            onChange={intent => onIntent(row.id, intent)}
            options={options}
            value={intents[row.id] ?? 'later'}
          />
        </div>
      ))}
    </div>
  )
}
