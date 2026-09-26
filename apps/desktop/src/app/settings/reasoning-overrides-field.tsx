import { REASONING_EFFORT_VALUES } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { CONTROL_TEXT } from './constants'

interface OverrideRow {
  model: string
  effort: string
}

/** Raw `agent.reasoning_overrides` (`{model: effort}`) into editor rows. */
function normalize(value: unknown): OverrideRow[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return []
  }

  return Object.entries(value as Record<string, unknown>)
    .filter(([model, effort]) => model.trim() && typeof effort === 'string' && effort)
    .map(([model, effort]) => ({ model: model.trim(), effort: String(effort) }))
}

function rowsEqual(a: OverrideRow[], b: OverrideRow[]): boolean {
  return a.length === b.length && a.every((row, index) => row.model === b[index].model && row.effort === b[index].effort)
}

/**
 * Structured editor for `agent.reasoning_overrides` — a per-model reasoning-effort
 * map the config file already supports (`per-model override > global
 * reasoning_effort`) but the desktop surface could only hand-edit as YAML (#117908).
 * Rows are model id + the same effort value set the global picker uses; an entry
 * only persists once it has a model id, so autosave never writes blanks.
 */
export function ReasoningOverridesField({
  value,
  onChange
}: {
  value: unknown
  onChange: (next: Record<string, string>) => void
}) {
  const { t } = useI18n()
  const m = t.settings.model

  const [rows, setRows] = useState<OverrideRow[]>(() => normalize(value))
  // Last complete map we emitted (or seeded). Autosave echoes it back through
  // `value`; ignore that echo so in-progress rows stay (#117908 mirrors the
  // FallbackModelsField contract).
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see FallbackModelsField)
  const lastEmittedRef = useRef<OverrideRow[]>(normalize(value))

  useEffect(() => {
    const persisted = normalize(value)

    if (rowsEqual(persisted, lastEmittedRef.current)) {
      return
    }

    lastEmittedRef.current = persisted
    setRows(persisted)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  const commit = (next: OverrideRow[]) => {
    const emitted: Record<string, string> = {}

    for (const row of next) {
      if (row.model.trim()) {
        emitted[row.model.trim()] = row.effort
      }
    }

    setRows(next)
    lastEmittedRef.current = Object.entries(emitted).map(([model, effort]) => ({ model, effort }))
    onChange(emitted)
  }

  const updateRow = (index: number, patch: Partial<OverrideRow>) =>
    commit(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)))

  const effortLabel = (effort: string) => t.shell.modelOptions[effort as keyof typeof t.shell.modelOptions] ?? effort

  return (
    <div className="grid w-full gap-1.5" data-testid="reasoning-overrides-field">
      {rows.map((row, index) => {
        const options: string[] = REASONING_EFFORT_VALUES.includes(row.effort as never)
          ? [...REASONING_EFFORT_VALUES]
          : [...REASONING_EFFORT_VALUES, row.effort]

        return (
          <div className="flex flex-wrap items-center gap-2" key={index}>
            <Input
              aria-label={`Model ${index + 1}`}
              className="min-w-52 flex-1"
              onChange={event => updateRow(index, { model: event.target.value })}
              placeholder="model id"
              value={row.model}
            />
            <Select onValueChange={effort => updateRow(index, { effort })} value={row.effort}>
              <SelectTrigger className={cn('min-w-36', CONTROL_TEXT)}>
                <SelectValue>{effortLabel(row.effort)}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {options.map(effort => (
                  <SelectItem key={effort} value={effort}>
                    {effortLabel(effort)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              aria-label={t.common.remove}
              onClick={() => commit(rows.filter((_, i) => i !== index))}
              size="icon-xs"
              variant="ghost"
            >
              <X className="size-3.5" />
            </Button>
          </div>
        )
      })}
      <div>
        <Button onClick={() => commit([...rows, { model: '', effort: 'medium' }])} size="sm" variant="textStrong">
          <Plus className="size-3.5" />
          {m.overrideAdd}
        </Button>
      </div>
    </div>
  )
}
