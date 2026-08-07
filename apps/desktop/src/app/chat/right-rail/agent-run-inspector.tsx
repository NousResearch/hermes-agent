import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

interface AgentRunInspectorProps {
  awaitingResponse: boolean
  busy: boolean
  gatewayOpen: boolean
  gatewayState: string
  messageCount: number
  model: string | null | undefined
  provider: string | null | undefined
  sessionId: string | null
}

type Tone = 'emerald' | 'amber' | 'blue' | 'muted'

function toneClasses(tone: Tone) {
  switch (tone) {
    case 'amber':
      return 'border-amber-500/35 bg-amber-500/10 text-amber-300'

    case 'blue':
      return 'border-sky-500/35 bg-sky-500/10 text-sky-300'

    case 'emerald':
      return 'border-emerald-500/35 bg-emerald-500/10 text-emerald-300'

    case 'muted':
      return 'border-(--ui-stroke-tertiary) bg-(--ui-bg-primary)/45 text-(--ui-text-tertiary)'
  }
}

function StatusDot({ active = true, tone = 'emerald' }: { active?: boolean; tone?: Tone }) {
  const color = !active
    ? 'bg-(--ui-text-quaternary)'
    : tone === 'amber'
      ? 'bg-amber-400'
      : tone === 'blue'
        ? 'bg-sky-400'
        : 'bg-emerald-400'

  return <span aria-hidden="true" className={cn('size-1.5 shrink-0 rounded-full', color)} />
}

function Pill({ children, tone = 'muted' }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.6875rem]', toneClasses(tone))}>
      {children}
    </span>
  )
}

function InspectorCard({
  children,
  icon,
  right,
  title
}: {
  children: ReactNode
  icon: string
  right?: ReactNode
  title: string
}) {
  return (
    <section className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/60 p-3 shadow-sm shadow-black/10">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
          <Codicon name={icon} size="0.8rem" />
          <span className="truncate">{title}</span>
        </div>
        {right}
      </div>
      {children}
    </section>
  )
}

function MetadataRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[0.75rem]">
      <span className="shrink-0 text-(--ui-text-tertiary)">{label}</span>
      <span className="min-w-0 truncate text-right font-medium text-(--ui-text-primary)">{value}</span>
    </div>
  )
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 overflow-hidden rounded-full bg-(--ui-bg-primary)">
      <div className="h-full rounded-full bg-emerald-400 transition-[width]" style={{ width: `${value}%` }} />
    </div>
  )
}

function ToolRow({ active = true, label, meta }: { active?: boolean; label: string; meta: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-(--ui-stroke-quaternary) bg-(--ui-bg-primary)/35 px-2 py-1.5 text-[0.72rem]">
      <span className="flex min-w-0 items-center gap-2">
        <StatusDot active={active} tone={active ? 'emerald' : 'muted'} />
        <span className="truncate text-(--ui-text-primary)">{label}</span>
      </span>
      <span className="shrink-0 text-(--ui-text-tertiary)">{meta}</span>
    </div>
  )
}

function TimelineRow({
  active = true,
  label,
  meta,
  tone = 'emerald'
}: {
  active?: boolean
  label: string
  meta: string
  tone?: Tone
}) {
  return (
    <li className="grid grid-cols-[0.75rem_1fr_auto] items-center gap-2 text-[0.72rem]">
      <StatusDot active={active} tone={tone} />
      <span className="min-w-0 truncate text-(--ui-text-primary)">{label}</span>
      <span className="text-(--ui-text-tertiary)">{meta}</span>
    </li>
  )
}

export function AgentRunInspector({
  awaitingResponse,
  busy,
  gatewayOpen,
  gatewayState,
  messageCount,
  model,
  provider,
  sessionId
}: AgentRunInspectorProps) {
  const runActive = busy || awaitingResponse
  const statusTone: Tone = !gatewayOpen ? 'amber' : runActive ? 'blue' : 'emerald'
  const statusLabel = !gatewayOpen ? 'Gateway closed' : runActive ? 'Running' : 'Healthy'
  const conversationLoad = Math.min(100, Math.max(8, messageCount * 8))
  const visibleSessionId = sessionId ? sessionId.slice(0, 8) : 'New chat'

  return (
    <aside className="hidden h-full w-[22rem] shrink-0 border-l border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) text-(--ui-text-secondary) xl:flex xl:flex-col">
      <div className="flex h-(--titlebar-height) shrink-0 items-center justify-between border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) px-3">
        <div className="flex min-w-0 items-center gap-2">
          <div className="grid size-6 shrink-0 place-items-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)">
            <Codicon className="text-(--ui-text-tertiary)" name="dashboard" size="0.9rem" />
          </div>
          <div className="min-w-0">
            <h3 className="truncate text-[0.75rem] font-semibold text-(--ui-text-primary)">Agent Run Inspector</h3>
            <p className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">Live Desktop run visibility</p>
          </div>
        </div>
        <Pill tone={statusTone}>
          <StatusDot tone={statusTone} />
          {statusLabel}
        </Pill>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <InspectorCard
          icon="server-process"
          right={
            <Pill tone={provider ? 'emerald' : 'muted'}>
              <StatusDot active={Boolean(provider)} tone={provider ? 'emerald' : 'muted'} />
              {provider ? 'Selected' : 'Pending'}
            </Pill>
          }
          title="Provider / Model"
        >
          <div className="space-y-2">
            <MetadataRow label="Provider" value={provider || 'Not selected'} />
            <MetadataRow label="Model" value={model || 'Not selected'} />
            <MetadataRow label="Session" value={visibleSessionId} />
          </div>
        </InspectorCard>

        <InspectorCard
          icon="radio-tower"
          right={
            <Pill tone={gatewayOpen ? 'emerald' : 'amber'}>
              <StatusDot tone={gatewayOpen ? 'emerald' : 'amber'} />
              {gatewayOpen ? 'Connected' : 'Offline'}
            </Pill>
          }
          title="Execution Target"
        >
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2 py-2 text-center text-[0.72rem] font-medium text-emerald-300">
                Local Active
              </div>
              <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-primary)/35 px-2 py-2 text-center text-[0.72rem] text-(--ui-text-tertiary)">
                Remote Idle
              </div>
            </div>
            <MetadataRow label="Gateway state" value={gatewayState} />
          </div>
        </InspectorCard>

        <InspectorCard icon="graph-line" right={<span className="text-[0.6875rem] text-(--ui-text-tertiary)">{conversationLoad}%</span>} title="Context Signal">
          <div className="space-y-2">
            <ProgressBar value={conversationLoad} />
            <MetadataRow label="Visible messages" value={messageCount} />
            <div className="rounded-lg border border-(--ui-stroke-quaternary) bg-(--ui-bg-primary)/45 p-2 text-[0.72rem] leading-relaxed text-(--ui-text-tertiary)">
              Token usage metadata is not exposed yet. This slot is ready for prompt, completion, and context-window usage once the gateway emits it.
            </div>
          </div>
        </InspectorCard>

        <InspectorCard icon="tools" right={<span className="text-[0.6875rem] text-(--ui-text-tertiary)">4 visible</span>} title="Tools Loaded">
          <div className="space-y-1.5">
            <ToolRow active={gatewayOpen} label="Gateway tools" meta={gatewayOpen ? 'ready' : 'blocked'} />
            <ToolRow active={gatewayOpen} label="Model controls" meta={model ? 'ready' : 'pending'} />
            <ToolRow active={gatewayOpen} label="Session state" meta={sessionId ? 'active' : 'new'} />
            <ToolRow active={false} label="MCP/tool call stream" meta="next" />
          </div>
        </InspectorCard>

        <InspectorCard icon="checklist" title="Run Timeline">
          <ol className="space-y-2">
            <TimelineRow label="Chat view mounted" meta="done" />
            <TimelineRow active={gatewayOpen} label={gatewayOpen ? 'Gateway connected' : 'Waiting for gateway'} meta={gatewayOpen ? 'done' : 'wait'} tone={gatewayOpen ? 'emerald' : 'amber'} />
            <TimelineRow active={Boolean(provider || model)} label="Provider/model resolved" meta={provider || model ? 'done' : 'wait'} tone={provider || model ? 'emerald' : 'muted'} />
            <TimelineRow active={runActive} label={runActive ? 'Agent response in progress' : 'Idle'} meta={runActive ? 'live' : 'ready'} tone={runActive ? 'blue' : 'muted'} />
          </ol>
        </InspectorCard>

        <InspectorCard
          icon="pulse"
          right={
            <Pill tone={statusTone}>
              <StatusDot tone={statusTone} />
              {statusLabel}
            </Pill>
          }
          title="Run Status"
        >
          <div className="grid grid-cols-2 gap-2 text-[0.72rem]">
            <div className="rounded-lg border border-(--ui-stroke-quaternary) bg-(--ui-bg-primary)/35 p-2">
              <div className="text-(--ui-text-tertiary)">Responding</div>
              <div className="mt-1 font-medium text-(--ui-text-primary)">{awaitingResponse ? 'Yes' : 'No'}</div>
            </div>
            <div className="rounded-lg border border-(--ui-stroke-quaternary) bg-(--ui-bg-primary)/35 p-2">
              <div className="text-(--ui-text-tertiary)">Busy</div>
              <div className="mt-1 font-medium text-(--ui-text-primary)">{busy ? 'Yes' : 'No'}</div>
            </div>
          </div>
        </InspectorCard>
      </div>
    </aside>
  )
}
