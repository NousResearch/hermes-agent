/**
 * Projects — the project-centric companion to the Kanban board. Lists every
 * first-class Hermes project with live status / % complete / ETA, and drills
 * into one: where it lives, its lead, the stage funnel, a completion bar, the
 * plan's milestones + ETA, what's waiting on a human, the team carrying it,
 * recent activity, velocity, and a composer that routes a request to the lead.
 *
 * All data comes from the kanban plugin router via `ctx.rest` — `/projects/overview`,
 * `/projects/{ref}/overview`, and `/projects/{ref}/feedback`.
 *
 * Mounted at `/projects` (ROUTES_AREA) + a sidebar row + a palette command.
 */

import {
  Button,
  cn,
  Codicon,
  compactNumber,
  ErrorState,
  host,
  Input,
  Loader,
  ScrollArea,
  Separator,
  Textarea,
  Tip,
  useMutation,
  useQuery,
  useQueryClient
} from '@hermes/plugin-sdk'
import { type ReactNode, useState } from 'react'

import {
  $boardSlug,
  fetchProjectOverview,
  fetchProjectsOverview,
  projectsKey,
  projectsOverviewKey,
  projectOverviewKey,
  sendProjectFeedback,
  updateProject,
  useKanbanScope
} from './api'
import { columnLabel, useKanban } from './i18n'
import {
  columnMeta,
  type ProjectOverview,
  type ProjectStatus,
  type ProjectSummary
} from './types'
import { ago, Avatar, errText } from './ui'

// ── status presentation ──────────────────────────────────────────────────────

const STATUS_TONE: Record<ProjectStatus, string> = {
  active: '#34d399',
  blocked: '#f87171',
  complete: '#60a5fa',
  idle: 'var(--ui-text-tertiary)',
  no_work: 'var(--ui-text-quaternary)',
  unlinked: 'var(--ui-text-quaternary)'
}

function StatusBadge({ status }: { status: ProjectStatus }) {
  const k = useKanban()

  return (
    <span
      className="inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[0.625rem] font-medium"
      style={{
        backgroundColor: `color-mix(in srgb, ${STATUS_TONE[status]} 16%, transparent)`,
        color: STATUS_TONE[status]
      }}
    >
      <span className="size-1.5 rounded-full" style={{ backgroundColor: STATUS_TONE[status] }} />
      {k.proj.st[status]}
    </span>
  )
}

function ProgressBar({ percent, tone }: { percent: null | number; tone: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-quaternary)">
      <div
        className="h-full rounded-full transition-[width]"
        style={{ backgroundColor: tone, width: `${Math.max(0, Math.min(100, percent ?? 0))}%` }}
      />
    </div>
  )
}

function MetaRow({ children, icon }: { children: ReactNode; icon: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5">
      <Codicon className="shrink-0 text-(--ui-text-tertiary)" name={icon} size="0.8rem" />
      <span className="min-w-0 truncate">{children}</span>
    </span>
  )
}

function SectionCard({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="flex flex-col gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-3">
      <h2 className="text-[0.6875rem] font-semibold uppercase tracking-[0.1em] text-(--ui-text-tertiary)">{title}</h2>
      {children}
    </section>
  )
}

const emptyNote = 'text-[0.75rem] text-(--ui-text-quaternary)'

// Attention-first ordering: blocked projects lead, then things in flight, then
// the quiet ones (idle → complete → no_work → unlinked).
const STATUS_ORDER: Record<ProjectStatus, number> = {
  blocked: 0,
  active: 1,
  idle: 2,
  complete: 3,
  no_work: 4,
  unlinked: 5
}

// ── list ─────────────────────────────────────────────────────────────────────

function ProjectRow({ onOpen, project }: { onOpen: () => void; project: ProjectSummary }) {
  const k = useKanban()
  const tone = project.color || STATUS_TONE[project.status]

  return (
    <button
      className={cn(
        'flex w-full items-center gap-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-3 py-2.5 text-left',
        'transition-colors hover:bg-primary/[0.06]'
      )}
      onClick={onOpen}
      type="button"
    >
      <span
        className="grid size-8 shrink-0 place-items-center rounded-md"
        style={{ backgroundColor: `color-mix(in srgb, ${project.color || 'var(--ui-text-tertiary)'} 14%, transparent)` }}
      >
        <Codicon name={project.icon || 'project'} size="1rem" />
      </span>

      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex items-center gap-2">
          <span className="truncate text-[0.8125rem] font-medium text-foreground">{project.name}</span>
          <StatusBadge status={project.status} />
        </span>
        {project.primary_path && (
          <span className="min-w-0 truncate font-mono text-[0.625rem] text-(--ui-text-quaternary)">
            {project.primary_path}
          </span>
        )}
      </span>

      {/* progress — the at-a-glance completion signal */}
      <span className="hidden w-40 shrink-0 flex-col gap-1 sm:flex">
        <span className="flex items-center justify-between text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
          <span>
            {project.percent_complete != null
              ? k.proj.complete(project.percent_complete)
              : project.board_slug
                ? k.proj.noProgress
                : k.proj.noBoard}
          </span>
          <span>{k.proj.tasks(project.totals.total)}</span>
        </span>
        <ProgressBar percent={project.percent_complete} tone={tone} />
      </span>

      <span className="flex shrink-0 items-center gap-2 text-[0.6875rem] text-(--ui-text-tertiary)">
        {project.eta ? (
          <Tip label={k.proj.eta}>
            <span className="inline-flex items-center gap-1 rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5">
              <Codicon name="calendar" size="0.7rem" />
              {project.eta}
            </span>
          </Tip>
        ) : null}
        {project.lead ? (
          <span className="inline-flex items-center gap-1">
            <Avatar name={project.lead} size="1rem" />
            {project.lead}
          </span>
        ) : (
          <span className="text-(--ui-text-quaternary)">{k.proj.noLead}</span>
        )}
        <Codicon name="chevron-right" size="0.9rem" />
      </span>
    </button>
  )
}

function ProjectList({ onOpen }: { onOpen: (ref: string) => void }) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const { data, error } = useQuery({
    queryKey: projectsOverviewKey(scope),
    queryFn: fetchProjectsOverview,
    refetchInterval: 60_000
  })
  const projects = data?.projects ?? []
  const sorted = [...projects].sort(
    (a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status] || b.totals.total - a.totals.total
  )
  const totalCards = projects.reduce((sum, p) => sum + p.totals.total, 0)
  const totalBlocked = projects.reduce((sum, p) => sum + p.totals.blocked, 0)

  return (
    <div className="flex h-full flex-col overflow-hidden bg-(--ui-surface-background)">
      <header className="flex shrink-0 items-start gap-2 px-4 pt-3 pb-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h1 className="text-sm font-semibold text-foreground">{k.proj.title}</h1>
          <p className="text-[0.6875rem] text-(--ui-text-tertiary)">{k.proj.subtitle}</p>
          {projects.length > 0 && (
            <p className="text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)">
              {k.proj.summaryLine(projects.length, totalCards, totalBlocked)}
            </p>
          )}
        </div>
        <Tip label={k.proj.refresh}>
          <Button
            aria-label={k.proj.refresh}
            className="ml-auto"
            onClick={() => void qc.invalidateQueries({ queryKey: projectsOverviewKey(scope) })}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="refresh" size="0.85rem" />
          </Button>
        </Tip>
      </header>
      <ScrollArea className="flex-1 px-4 pb-4">
        {error && !data ? (
          <div className="grid h-full place-items-center">
            <ErrorState title={errText(error)} />
          </div>
        ) : !data ? (
          <div className="grid h-full place-items-center">
            <Loader type="lemniscate-bloom" />
          </div>
        ) : projects.length === 0 ? (
          <div className="grid h-full place-items-center px-4 text-center">
            <div className="flex max-w-sm flex-col items-center gap-2">
              <Codicon className="text-(--ui-text-quaternary)" name="project" size="1.25rem" />
              <p className="text-xs text-(--ui-text-tertiary)">{k.proj.empty}</p>
              <p className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.proj.emptyHint}</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {sorted.map(project => (
              <ProjectRow key={project.id} onOpen={() => onOpen(project.id)} project={project} />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

// ── detail panels ────────────────────────────────────────────────────────────

function Pipeline({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()
  const stages = overview.stages.filter(stage => stage.count > 0)

  return (
    <SectionCard title={k.proj.pipeline}>
      {overview.totals.total === 0 ? (
        <span className={emptyNote}>{k.proj.noProgress}</span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {stages.map(stage => (
            <span
              className="inline-flex items-center gap-1.5 rounded-md bg-(--ui-bg-quaternary) px-2 py-1 text-[0.6875rem] text-(--ui-text-secondary)"
              key={stage.name}
              title={columnLabel(k, stage.name)}
            >
              <span className="size-1.5 rounded-full" style={{ backgroundColor: columnMeta(stage.name).tone }} />
              {columnLabel(k, stage.name)}
              <span className="tabular-nums text-(--ui-text-tertiary)">{stage.count}</span>
            </span>
          ))}
          {overview.totals.awaiting ? (
            <span
              className="inline-flex items-center gap-1.5 rounded-md bg-(--ui-bg-quaternary) px-2 py-1 text-[0.6875rem] text-(--ui-text-secondary)"
              title={k.proj.awaitingPublication}
            >
              <span className="size-1.5 rounded-full" style={{ backgroundColor: '#a78bfa' }} />
              {k.proj.awaitingPublication}
              <span className="tabular-nums text-(--ui-text-tertiary)">{overview.totals.awaiting}</span>
            </span>
          ) : null}
        </div>
      )}
      <span className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.proj.tasks(overview.totals.total)}</span>
    </SectionCard>
  )
}

function Attention({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()

  return (
    <SectionCard title={k.proj.attention}>
      {overview.attention.length === 0 ? (
        <span className={emptyNote}>{k.proj.noAttention}</span>
      ) : (
        <div className="flex flex-col gap-1.5">
          {overview.attention.map(item => (
            <div className="flex items-center gap-2 text-[0.75rem]" key={item.id}>
              <Codicon className="shrink-0 text-[#f87171]" name="error" size="0.8rem" />
              <span className="min-w-0 flex-1 truncate text-(--ui-text-secondary)" title={item.title}>
                {item.title}
              </span>
              {item.assignee && <Avatar name={item.assignee} size="1rem" />}
              {ago(item.created_at) && (
                <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{ago(item.created_at)}</span>
              )}
            </div>
          ))}
          <span className="text-[0.625rem] text-(--ui-text-quaternary)">{k.proj.attentionHint}</span>
        </div>
      )}
    </SectionCard>
  )
}

function Team({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()

  return (
    <SectionCard title={k.proj.team}>
      {overview.team.length === 0 ? (
        <span className={emptyNote}>{k.proj.noTeam}</span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {overview.team.map(member => (
            <span
              className="inline-flex items-center gap-1.5 rounded-md bg-(--ui-bg-quaternary) px-2 py-1 text-[0.6875rem] text-(--ui-text-secondary)"
              key={member.name}
            >
              <Avatar name={member.name} size="1rem" />
              {member.name}
              {member.running > 0 && (
                <span className="inline-flex items-center gap-0.5 text-[#34d399]" title={columnLabel(k, 'running')}>
                  <Codicon name="sync" size="0.65rem" spinning />
                  {member.running}
                </span>
              )}
              <span className="tabular-nums text-(--ui-text-tertiary)">{member.total}</span>
            </span>
          ))}
        </div>
      )}
    </SectionCard>
  )
}

function Activity({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()

  return (
    <SectionCard title={k.proj.activity}>
      {overview.activity.length === 0 ? (
        <span className={emptyNote}>{k.proj.noActivity}</span>
      ) : (
        <div className="flex flex-col gap-1">
          {overview.activity.map((event, i) => (
            <div className="flex items-center gap-2 text-[0.6875rem]" key={`${event.task_id}-${event.created_at}-${i}`}>
              <span className="shrink-0 rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5 font-mono text-[0.625rem] text-(--ui-text-tertiary)">
                {event.kind}
              </span>
              <span className="min-w-0 flex-1 truncate text-(--ui-text-secondary)" title={event.title}>
                {event.title || event.task_id}
              </span>
              {ago(event.created_at) && (
                <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{ago(event.created_at)}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </SectionCard>
  )
}

function Plan({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()
  const plan = overview.plan

  return (
    <SectionCard title={k.proj.plan}>
      {!plan.exists ? (
        <span className={emptyNote}>{k.proj.noPlan(plan.path || overview.location)}</span>
      ) : (
        <div className="flex flex-col gap-1.5">
          {plan.total_count > 0 && (
            <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
              {k.proj.milestones(plan.done_count, plan.total_count)}
              {plan.percent != null ? ` · ${k.proj.complete(plan.percent)}` : ''}
            </span>
          )}
          {plan.milestones.map((milestone, i) => (
            <span className="flex items-center gap-2 text-[0.75rem]" key={`${milestone.title}-${i}`}>
              <Codicon
                className={milestone.done ? 'text-[#34d399]' : 'text-(--ui-text-quaternary)'}
                name={milestone.done ? 'pass-filled' : 'circle-large-outline'}
                size="0.8rem"
              />
              <span className={cn('min-w-0 flex-1 truncate', milestone.done && 'text-(--ui-text-tertiary) line-through')}>
                {milestone.title}
              </span>
              {milestone.date && (
                <span className="shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
                  {k.proj.due(milestone.date)}
                </span>
              )}
            </span>
          ))}
        </div>
      )}
    </SectionCard>
  )
}

function Velocity({ overview }: { overview: ProjectOverview }) {
  const k = useKanban()
  const v = overview.velocity

  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[0.6875rem] text-(--ui-text-tertiary)">
      <span>{k.proj.done7d(compactNumber(v.done_7d))}</span>
      {v.per_day ? <span>{k.proj.perDay(v.per_day)}</span> : null}
      {v.projected_finish ? <span className="text-(--ui-text-secondary)">{k.proj.projected(v.projected_finish)}</span> : null}
    </span>
  )
}

function FeedbackComposer({ overview, projectRef }: { overview: ProjectOverview; projectRef: string }) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const [body, setBody] = useState('')
  const hasLead = Boolean(overview.lead)

  const send = useMutation({
    mutationFn: () => sendProjectFeedback(projectRef, body.trim()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      setBody('')
      host.notify({ kind: 'success', message: k.proj.feedbackSent })
      void qc.invalidateQueries({ queryKey: projectOverviewKey(scope, projectRef) })
    }
  })

  return (
    <SectionCard title={k.proj.feedback}>
      <Textarea
        className="min-h-16"
        onChange={event => setBody(event.target.value)}
        placeholder={k.proj.feedbackPlaceholder}
        value={body}
      />
      <span className="text-[0.6875rem] leading-relaxed text-(--ui-text-quaternary)">
        {hasLead ? k.proj.feedbackHint : k.proj.feedbackHintNoLead}
      </span>
      <div className="flex items-center justify-between gap-2">
        {hasLead ? (
          <span className="inline-flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
            <Avatar name={overview.lead} size="1rem" />
            {overview.lead}
          </span>
        ) : (
          <span />
        )}
        <Button disabled={!body.trim() || send.isPending} onClick={() => send.mutate()} size="sm">
          <Codicon name={send.isPending ? 'loading' : 'send'} size="0.75rem" spinning={send.isPending} />
          {k.proj.send}
        </Button>
      </div>
    </SectionCard>
  )
}

function SettingsPanel({
  overview,
  projectRef,
  onClose
}: {
  overview: ProjectOverview
  projectRef: string
  onClose: () => void
}) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const [lead, setLead] = useState(overview.project.lead || '')
  const [planPath, setPlanPath] = useState(overview.project.plan_path || '')

  const save = useMutation({
    mutationFn: () => updateProject(projectRef, { lead: lead.trim(), plan_path: planPath.trim() }),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      host.notify({ kind: 'success', message: k.proj.saved })
      void qc.invalidateQueries({ queryKey: projectOverviewKey(scope, projectRef) })
      void qc.invalidateQueries({ queryKey: projectsKey(scope) })
      void qc.invalidateQueries({ queryKey: projectsOverviewKey(scope) })
      onClose()
    }
  })

  return (
    <SectionCard title={k.proj.settings}>
      <span className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.proj.settingsHint}</span>
      <label className="flex flex-col gap-1">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-(--ui-text-quaternary)">
          {k.proj.lead}
        </span>
        <Input onChange={event => setLead(event.target.value)} placeholder={k.proj.leadPlaceholder} value={lead} />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-(--ui-text-quaternary)">
          {k.proj.plan}
        </span>
        <Input
          onChange={event => setPlanPath(event.target.value)}
          placeholder={k.proj.planPathPlaceholder}
          value={planPath}
        />
      </label>
      <div className="flex justify-end gap-2">
        <Button onClick={onClose} size="sm" variant="text">
          {k.cancel}
        </Button>
        <Button disabled={save.isPending} onClick={() => save.mutate()} size="sm">
          {k.proj.save}
        </Button>
      </div>
    </SectionCard>
  )
}

function ProjectDetail({ onBack, projectRef }: { onBack: () => void; projectRef: string }) {
  const k = useKanban()
  const scope = useKanbanScope()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const { data: overview, error } = useQuery({
    queryFn: () => fetchProjectOverview(projectRef),
    queryKey: projectOverviewKey(scope, projectRef),
    refetchInterval: 60_000
  })

  const openBoard = () => {
    if (overview?.board?.slug) {
      $boardSlug.set(overview.board.slug)
    }
    host.navigate('/kanban')
  }

  if (error && !overview) {
    return (
      <div className="grid h-full place-items-center">
        <ErrorState title={errText(error)} />
      </div>
    )
  }

  if (!overview) {
    return (
      <div className="grid h-full place-items-center">
        <Loader type="lemniscate-bloom" />
      </div>
    )
  }

  const project = overview.project
  const pct = overview.percent_complete
  const tone = project.color || STATUS_TONE[overview.status]

  return (
    <div className="flex h-full flex-col overflow-hidden bg-(--ui-surface-background)">
      <header className="flex shrink-0 items-center gap-3 px-4 pt-3 pb-2">
        <Tip label={k.proj.back}>
          <Button aria-label={k.proj.back} onClick={onBack} size="icon-xs" variant="ghost">
            <Codicon name="arrow-left" size="0.9rem" />
          </Button>
        </Tip>
        <Codicon name={project.icon || 'project'} size="1.1rem" />
        <h1 className="min-w-0 truncate text-sm font-semibold text-foreground">{project.name}</h1>
        <StatusBadge status={overview.status} />
        <div className="ml-auto flex items-center gap-1.5">
          {overview.board && (
            <Button onClick={openBoard} size="sm" variant="outline">
              <Codicon name="project" size="0.75rem" />
              {k.proj.openBoard}
            </Button>
          )}
          <Button onClick={() => setSettingsOpen(!settingsOpen)} size="icon-xs" variant="ghost">
            <Codicon name="settings-gear" size="0.9rem" />
          </Button>
        </div>
      </header>

      <ScrollArea className="flex-1 px-4 pb-4">
        <div className="flex flex-col gap-2">
          {/* where it is + who owns it */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-1 text-[0.75rem] text-(--ui-text-secondary)">
            <MetaRow icon="folder">{overview.location || '—'}</MetaRow>
            <MetaRow icon="account">
              {overview.lead ? (
                <span className="inline-flex items-center gap-1.5">
                  <Avatar name={overview.lead} size="1rem" />
                  {overview.lead}
                </span>
              ) : (
                <span className="text-(--ui-text-quaternary)">{k.proj.noLead}</span>
              )}
            </MetaRow>
            <MetaRow icon="project">
              {overview.board ? (
                <span className="font-mono">{overview.board.name || overview.board.slug}</span>
              ) : (
                <span className="text-(--ui-text-quaternary)">{k.proj.noBoard}</span>
              )}
            </MetaRow>
          </div>

          {/* completion + eta + velocity */}
          <div className="grid gap-2 sm:grid-cols-2">
            <SectionCard title={k.proj.status}>
              <div className="flex items-baseline gap-2">
                {pct != null ? (
                  <>
                    <span className="text-2xl font-semibold tabular-nums text-foreground">{pct}%</span>
                    <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
                      {k.proj.tasks(overview.totals.total)}
                    </span>
                  </>
                ) : (
                  <span className="text-[0.75rem] text-(--ui-text-quaternary)">
                    {overview.board ? k.proj.noProgress : k.proj.noBoard}
                  </span>
                )}
              </div>
              <ProgressBar percent={pct} tone={tone} />
              <Velocity overview={overview} />
            </SectionCard>
            <SectionCard title={k.proj.eta}>
              <span
                className={cn('text-lg font-medium', overview.eta ? 'text-foreground' : 'text-(--ui-text-quaternary)')}
              >
                {overview.eta || k.proj.noEta}
              </span>
              {overview.plan.total_count > 0 && (
                <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
                  {k.proj.milestones(overview.plan.done_count, overview.plan.total_count)}
                </span>
              )}
            </SectionCard>
          </div>

          <Pipeline overview={overview} />
          <div className="grid gap-2 lg:grid-cols-2">
            <Attention overview={overview} />
            <Team overview={overview} />
          </div>
          <Plan overview={overview} />
          <Activity overview={overview} />
          {settingsOpen && (
            <SettingsPanel
              key="settings"
              onClose={() => setSettingsOpen(false)}
              overview={overview}
              projectRef={projectRef}
            />
          )}
          <Separator />
          <FeedbackComposer overview={overview} projectRef={projectRef} />
        </div>
      </ScrollArea>
    </div>
  )
}

export function KanbanProjectsPage() {
  const [selected, setSelected] = useState('')

  return selected ? (
    <ProjectDetail onBack={() => setSelected('')} projectRef={selected} />
  ) : (
    <ProjectList onOpen={setSelected} />
  )
}
