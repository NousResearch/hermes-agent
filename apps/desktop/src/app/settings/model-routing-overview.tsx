import { useI18n } from '@/i18n'
import { Layers3 } from '@/lib/icons'
import type { AuxiliaryModelsResponse } from '@/types/hermes'

import { Pill, SectionHeading } from './primitives'

// Mirrors `_AUX_TASK_SLOTS` in hermes_cli/web_server.py. Keeping the overview
// and editor on one list prevents a backend-supported lane from disappearing
// from either Desktop surface.
export const AUX_MODEL_TASK_KEYS = [
  'vision',
  'web_extract',
  'compression',
  'skills_hub',
  'approval',
  'mcp',
  'title_generation',
  'triage_specifier',
  'kanban_decomposer',
  'profile_describer',
  'curator'
] as const

export interface ModelRoutingLane {
  inherited: boolean
  key: 'main' | (typeof AUX_MODEL_TASK_KEYS)[number]
  model: string
  provider: string
}

export function buildModelRoutingLanes(
  mainModel: { model: string; provider: string } | null,
  auxiliary: AuxiliaryModelsResponse | null
): ModelRoutingLane[] {
  const main = mainModel ?? auxiliary?.main ?? { model: '', provider: '' }
  const assignments = new Map(auxiliary?.tasks.map(task => [task.task, task]) ?? [])

  const lanes: ModelRoutingLane[] = [
    {
      inherited: false,
      key: 'main',
      model: main.model,
      provider: main.provider
    }
  ]

  for (const key of AUX_MODEL_TASK_KEYS) {
    const assignment = assignments.get(key)
    const inherited = !assignment?.provider || assignment.provider === 'auto'

    lanes.push({
      inherited,
      key,
      model: inherited ? main.model : assignment.model,
      provider: inherited ? main.provider : assignment.provider
    })
  }

  return lanes
}

export function ModelRoutingOverview({
  auxiliary,
  mainModel
}: {
  auxiliary: AuxiliaryModelsResponse | null
  mainModel: { model: string; provider: string } | null
}) {
  const m = useI18n().t.settings.model
  const lanes = buildModelRoutingLanes(mainModel, auxiliary)

  return (
    <section aria-label={m.routingTitle}>
      <SectionHeading icon={Layers3} meta={m.routingLaneCount(lanes.length)} title={m.routingTitle} />
      <p className="mb-2.5 text-xs text-muted-foreground">{m.routingDesc}</p>
      <ol className="divide-y divide-(--ui-stroke-tertiary) overflow-hidden rounded-md border border-(--ui-stroke-secondary) bg-background/45">
        {lanes.map(lane => {
          const task = lane.key === 'main' ? null : m.tasks[lane.key]
          const label = lane.key === 'main' ? m.mainLane : (task?.label ?? lane.key)
          const hint = lane.key === 'main' ? m.mainLaneHint : (task?.hint ?? lane.key)

          return (
            <li className="@container" data-routing-lane key={lane.key}>
              <div className="grid min-h-11 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 px-3 py-2 @xl:grid-cols-[minmax(8rem,0.75fr)_minmax(0,1.5fr)_auto]">
                <div className="col-start-1 row-start-1 min-w-0">
                  <div className="truncate text-xs font-medium text-foreground">{label}</div>
                  <div className="truncate text-[0.68rem] text-(--ui-text-tertiary)">{hint}</div>
                </div>
                <div className="col-span-2 row-start-2 flex min-w-0 items-baseline gap-2 font-mono text-[0.68rem] @xl:col-span-1 @xl:col-start-2 @xl:row-start-1">
                  <span className="shrink-0 text-(--ui-text-secondary)">{lane.provider || m.routingUnavailable}</span>
                  <span aria-hidden="true" className="text-(--ui-text-tertiary)">
                    /
                  </span>
                  <span className="min-w-0 truncate text-(--ui-text-tertiary)" title={lane.model || undefined}>
                    {lane.model || m.providerDefault}
                  </span>
                </div>
                <div className="col-start-2 row-start-1 @xl:col-start-3">
                  <Pill tone={lane.inherited ? 'muted' : 'primary'}>
                    {lane.key === 'main' ? m.primary : lane.inherited ? m.usesMain : m.dedicated}
                  </Pill>
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
