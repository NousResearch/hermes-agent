import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getAutomationBlueprints } from '@/hermes'
import type { AutomationBlueprint, AutomationBlueprintField } from '@/hermes'
import { useI18n } from '@/i18n'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'

import { PanelDetail, PanelEmpty, PanelPill } from '../overlays/panel'

// The blueprint catalog is shared with the dashboard, so its deliver slot
// defaults to "origin" (the chat/home-channel a dashboard or gateway job was
// created from). Desktop has no origin chat, so seed the deliver slot to the
// desktop's native target ("local" = This desktop) instead. The dialog then
// renders that slot with the shared DeliverSelect (backend-sourced targets), so
// the raw "origin" option never reaches the desktop UI.
const DELIVER_FIELD = 'deliver'
const DESKTOP_DELIVER_DEFAULT = 'local'

function isDeliverField(field: AutomationBlueprintField): boolean {
  return field.name === DELIVER_FIELD
}

// Initial form state for a blueprint = each field's default (or ''). Pure so the
// suite can assert the form seeds correctly without mounting React. The deliver
// slot is special-cased: an "origin" default (or empty) becomes "local" so a
// desktop-created job delivers to This desktop instead of nowhere.
export function initialBlueprintValues(blueprint: AutomationBlueprint): Record<string, string> {
  const out: Record<string, string> = {}

  for (const field of blueprint.fields) {
    const seeded = field.default ?? ''
    out[field.name] = isDeliverField(field) && (seeded === '' || seeded === 'origin') ? DESKTOP_DELIVER_DEFAULT : seeded
  }

  return out
}

// A slot-level validation error from the backend arrives as "422: <message>"
// (or "<code>: <message>"); strip the leading numeric code for inline display.
export function cleanBlueprintFieldError(message: string): string {
  return message.replace(/^\d+:\s*/, '')
}

// Help text to show under a slot control. The backend deliver help is
// origin/dashboard-centric and even contradicts desktop semantics ("local =
// save only" vs. This desktop), and the DeliverSelect is self-explanatory —
// skip it for the deliver slot.
export function blueprintSlotHelp(field: AutomationBlueprintField): string | undefined {
  return field.help && field.type !== 'text' && !isDeliverField(field) ? field.help : undefined
}

// Renders one blueprint slot's control (enum/weekdays → Select, time → time
// input, else text). The deliver slot is handled separately by the dialog's
// shared DeliverSelect, so it's not rendered here.
export function BlueprintSlotControl({
  field,
  id,
  onChange,
  value
}: {
  field: AutomationBlueprintField
  id: string
  onChange: (next: string) => void
  value: string
}) {
  if (field.type === 'enum' || field.type === 'weekdays') {
    return (
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className="h-9 rounded-md" id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {field.options.map(option => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (field.type === 'time') {
    return <Input id={id} onChange={event => onChange(event.target.value)} type="time" value={value} />
  }

  return (
    <Input
      id={id}
      onChange={event => onChange(event.target.value)}
      placeholder={field.help || field.label}
      type="text"
      value={value}
    />
  )
}

// A clickable blueprint card — mirrors the app's other selectable cards
// (theme/pet/gateway/profile pickers) via selectableCardClass. Clicking opens
// the shared cron editor dialog pre-filled with this blueprint's slots; there's
// no inline expand form or divider.
function BlueprintCard({ blueprint, onSetUp }: { blueprint: AutomationBlueprint; onSetUp: () => void }) {
  return (
    <button
      className={cn(selectableCardClass({ prominent: true }), 'w-full p-2 text-left')}
      onClick={onSetUp}
      type="button"
    >
      <div className="min-w-0">
        <p className="text-sm font-medium text-foreground">{blueprint.title}</p>
        <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{blueprint.description}</p>
        {blueprint.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {blueprint.tags.map(tag => (
              <PanelPill key={tag}>{tag}</PanelPill>
            ))}
          </div>
        )}
      </div>
    </button>
  )
}

// Automation Blueprints gallery — the desktop counterpart to the dashboard's
// blueprint tab. Each card opens the shared cron editor dialog pre-filled with
// the blueprint's typed slots; submitting POSTs to
// /api/cron/blueprints/instantiate, which fills the blueprint and creates the
// job via the same create_job path as a hand-written cron.
export function BlueprintsPanel({ onSetUp }: { onSetUp: (blueprint: AutomationBlueprint) => void }) {
  const { t } = useI18n()
  const c = t.cron

  const blueprints = useQuery({
    queryKey: ['cron-blueprints'],
    queryFn: async () => (await getAutomationBlueprints()).blueprints
  })

  const cards = useMemo(() => blueprints.data ?? [], [blueprints.data])

  if (blueprints.isLoading) {
    return <PageLoader label={c.blueprints.loading} />
  }

  if (blueprints.isError) {
    return <PanelEmpty description={c.blueprints.failedLoad} icon="warning" title={c.blueprints.failedLoad} />
  }

  if (cards.length === 0) {
    return <PanelEmpty description={c.blueprints.emptyDesc} icon="lightbulb" title={c.blueprints.emptyTitle} />
  }

  return (
    <PanelDetail>
      {cards.map(blueprint => (
        <BlueprintCard blueprint={blueprint} key={blueprint.key} onSetUp={() => onSetUp(blueprint)} />
      ))}
    </PanelDetail>
  )
}
