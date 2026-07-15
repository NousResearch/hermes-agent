import { useStore } from '@nanostores/react'
import { lazy, Suspense, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { validateStudySchedule } from '@/lib/study-schemas'
import { cn } from '@/lib/utils'
import {
  $studyConfigured,
  $studyError,
  $studyInvalidSchedules,
  $studyLoadState,
  $studyMessage,
  $studyOverview,
  $studyProjects,
  $studySchedules,
  $studySelectedProjectId,
  $studySelectedSchedule,
  $studySelectedScheduleId,
  $studyVaultPath
} from '@/store/study'
import type { StudyProject, StudyScheduleEvent, StudySchedulePhase, StudyScheduleSummary } from '@/types/hermes'

import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import {
  cancelStudyActions,
  decideProposal,
  loadWorkspace,
  saveSettings,
  selectProject,
  selectSchedule,
  setStudyMessage
} from './actions'
import { httpStudyClient, type StudyClient } from './client'
import { StudyInbox } from './inbox'
import { StudySetup } from './setup'
import { StudyToday } from './today'

const ReviewView = lazy(async () => ({ default: (await import('./review')).ReviewView }))

interface StudyViewProps {
  client?: StudyClient
  onClose?: () => void
  onStartAgentReview?: (prompt: string) => void | Promise<void>
}

function formatDateTime(value: string): string {
  return value.slice(0, 16).replace('T', ' ')
}

function formatEffort(minutes: number): string {
  const hours = Math.floor(minutes / 60)
  const remainder = minutes % 60
  if (hours === 0) {
    return `${remainder}m`
  }
  return remainder === 0 ? `${hours}h` : `${hours}h ${remainder}m`
}

function monthKey(value: string): string {
  return value.slice(0, 7)
}

function weekLabel(value: string): string {
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  if (Number.isNaN(date.valueOf())) {
    return value.slice(0, 10)
  }
  const start = new Date(Date.UTC(date.getUTCFullYear(), 0, 1))
  const day = Math.floor((date.getTime() - start.getTime()) / 86_400_000) + 1
  return `W${Math.ceil(day / 7)}`
}

function subjectLabel(project: StudyProject | undefined, subjectId: string): string {
  const groups = project?.schema_version === 'study_project.v2' ? project.tracks : project?.subjects
  return groups?.find(subject => subject.id === subjectId)?.label ?? subjectId
}

function projectScopeLabel(project: StudyProject): string {
  if (project.schema_version === 'study_project.v2') {
    return `${project.domain} · ${project.deadline ?? project.phase}`
  }
  return `${project.exam_type ?? project.domain} · ${project.exam_date ?? project.phase}`
}

function eventLabel(project: StudyProject | undefined, event: StudyScheduleEvent): string {
  const label = subjectLabel(project, event.subject_id)
  return event.title.startsWith(label) ? event.title : `${label}：${event.title}`
}

function groupEvents(events: StudyScheduleEvent[]): Array<[string, Array<[string, StudyScheduleEvent[]]>]> {
  const months = new Map<string, Map<string, StudyScheduleEvent[]>>()
  for (const event of [...events].sort((a, b) => a.start.localeCompare(b.start))) {
    const month = monthKey(event.start)
    const week = weekLabel(event.start)
    if (!months.has(month)) {
      months.set(month, new Map())
    }
    const weeks = months.get(month)!
    weeks.set(week, [...(weeks.get(week) ?? []), event])
  }
  return [...months.entries()].map(([month, weeks]) => [month, [...weeks.entries()]])
}

function ProjectButton({
  active,
  project,
  onSelect
}: {
  active: boolean
  project: StudyProject
  onSelect: () => void
}) {
  return (
    <button
      className={cn(
        'w-full rounded-xl border px-3 py-3 text-left transition-colors',
        active
          ? 'border-primary/50 bg-primary/10 text-foreground'
          : 'border-border/60 bg-card/60 text-muted-foreground hover:bg-accent/50 hover:text-foreground'
      )}
      onClick={onSelect}
      type="button"
    >
      <div className="truncate text-sm font-semibold">{project.title}</div>
      <div className="mt-1 text-xs">{projectScopeLabel(project)}</div>
    </button>
  )
}

function ScheduleButton({
  active,
  schedule,
  onSelect
}: {
  active: boolean
  schedule: StudyScheduleSummary
  onSelect: () => void
}) {
  const { t } = useI18n()
  return (
    <button
      className={cn(
        'w-full rounded-xl border px-3 py-3 text-left transition-colors',
        active
          ? 'border-sky-500/60 bg-sky-500/10 text-foreground'
          : 'border-border/60 bg-card/60 text-muted-foreground hover:bg-accent/50 hover:text-foreground'
      )}
      onClick={onSelect}
      type="button"
    >
      <div className="truncate text-sm font-semibold">{schedule.title}</div>
      <div className="mt-1 text-xs">
        {schedule.range.start} → {schedule.range.end}
      </div>
      <div className="mt-1 text-xs">
        {schedule.phase_count ?? 0} {t.study.phases} · {schedule.event_count} {t.study.events}
      </div>
    </button>
  )
}

export function StudyView({ client = httpStudyClient, onClose, onStartAgentReview }: StudyViewProps) {
  const { t } = useI18n()
  const projects = useStore($studyProjects)
  const schedules = useStore($studySchedules)
  const invalidSchedules = useStore($studyInvalidSchedules)
  const selectedProjectId = useStore($studySelectedProjectId)
  const selectedScheduleId = useStore($studySelectedScheduleId)
  const selectedSchedule = useStore($studySelectedSchedule)
  const loadState = useStore($studyLoadState)
  const error = useStore($studyError)
  const configured = useStore($studyConfigured)
  const message = useStore($studyMessage)
  const overview = useStore($studyOverview)
  const vaultPath = useStore($studyVaultPath)
  const selectedProject = projects.find(project => project.project_id === selectedProjectId)
  const scheduleValidation = selectedSchedule ? validateStudySchedule(selectedSchedule) : null
  const [activeTab, setActiveTab] = useState<'today' | 'calendar' | 'review' | 'inbox' | 'settings'>('today')

  useEffect(() => {
    void loadWorkspace(client)
    return cancelStudyActions
  }, [client])

  const saveVaultSettings = async (nextVaultPath: string) => {
    const result = await saveSettings(client, nextVaultPath)
    setStudyMessage(result.requires_new_session ? t.study.newSessionRequired : t.study.settingsSaved)
    setActiveTab('today')
    return result
  }

  return (
    <OverlayView closeLabel={t.study.close} onClose={onClose ?? (() => undefined)}>
      <OverlaySplitLayout>
        <OverlaySidebar>
          <div className="space-y-5 p-4 pt-[calc(var(--titlebar-height)+1rem)]">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">{t.study.readOnly}</div>
              <h1 className="mt-2 text-2xl font-semibold">{t.study.title}</h1>
              <p className="mt-1 text-sm text-muted-foreground">{t.study.subtitle}</p>
            </div>
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {t.study.projectList}
              </div>
              {projects.map(project => (
                <ProjectButton
                  active={project.project_id === selectedProjectId}
                  key={project.project_id}
                  onSelect={() => void selectProject(client, project.project_id)}
                  project={project}
                />
              ))}
              {configured && projects.length === 0 && (
                <div className="text-sm text-muted-foreground">{t.study.noProjects}</div>
              )}
            </div>
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {t.study.scheduleList}
              </div>
              {schedules.map(schedule => (
                <ScheduleButton
                  active={schedule.schedule_id === selectedScheduleId}
                  key={schedule.schedule_id}
                  onSelect={() =>
                    selectedProjectId && void selectSchedule(client, selectedProjectId, schedule.schedule_id)
                  }
                  schedule={schedule}
                />
              ))}
              {invalidSchedules.length > 0 && (
                <div className="space-y-2 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3">
                  <div className="text-xs font-semibold text-amber-700 dark:text-amber-300">
                    {t.study.invalidScheduleFiles}
                  </div>
                  {invalidSchedules.map(schedule => (
                    <div className="space-y-1 text-xs" key={schedule.path}>
                      <div className="break-words font-medium">{schedule.schedule_id}</div>
                      <div className="break-all text-muted-foreground">{schedule.path}</div>
                      <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
                        {schedule.errors.map(error => (
                          <li key={error}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
              {selectedProjectId && schedules.length === 0 && invalidSchedules.length === 0 && (
                <div className="text-sm text-muted-foreground">{t.study.noSchedules}</div>
              )}
            </div>
            <Button className="w-full" onClick={() => setActiveTab('settings')} size="sm" variant="secondary">
              {t.study.settings}
            </Button>
          </div>
        </OverlaySidebar>
        <OverlayMain className="overflow-y-auto p-6 pt-[calc(var(--titlebar-height)+1.5rem)]">
          {loadState === 'loading' && !selectedSchedule && <PageLoader label={t.study.loading} />}
          {!configured && <StudySetup initialPath={vaultPath ?? ''} onSave={saveVaultSettings} />}
          {error && (
            <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-5 text-sm">
              <div className="font-semibold">{error}</div>
              <Button className="mt-4" onClick={() => void loadWorkspace(client)} size="sm" variant="secondary">
                {t.study.retry}
              </Button>
            </div>
          )}
          {configured && message && (
            <div className="mb-4 rounded-xl border border-primary/30 bg-primary/10 p-3 text-sm">{message}</div>
          )}
          {configured && activeTab === 'settings' && (
            <StudySetup initialPath={vaultPath ?? ''} onSave={saveVaultSettings} />
          )}
          {configured && projects.length > 0 && !selectedProject && activeTab !== 'settings' && (
            <div className="rounded-2xl border border-dashed bg-card/70 p-8 text-sm text-muted-foreground">
              {t.study.selectProjectPrompt}
            </div>
          )}
          {selectedProject && activeTab !== 'settings' && (
            <div className="mb-6 rounded-3xl border bg-[radial-gradient(circle_at_top_left,color-mix(in_srgb,var(--primary)_20%,transparent),transparent_32%),var(--card)] p-6 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-primary">
                    {selectedProject.domain_pack}
                  </div>
                  <h2 className="mt-2 text-3xl font-semibold tracking-tight">{selectedProject.title}</h2>
                </div>
                <div className="rounded-full border bg-background/70 px-3 py-1 text-xs font-medium text-muted-foreground">
                  {t.study.readOnly}
                </div>
              </div>
              <div className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
                <Meta
                  label={selectedProject.schema_version === 'study_project.v2' ? t.study.deadline : t.study.examDate}
                  value={
                    selectedProject.schema_version === 'study_project.v2'
                      ? (selectedProject.deadline ?? '—')
                      : selectedProject.exam_date
                  }
                />
                <Meta label={t.study.phase} value={selectedProject.phase} />
                <Meta label={t.study.timezone} value={selectedProject.timezone} />
              </div>
            </div>
          )}
          {selectedProject && activeTab !== 'settings' && (
            <div className="mb-4 flex gap-1">
              {(
                [
                  ['today', t.study.todayTab],
                  ['calendar', t.study.calendarTab],
                  ['review', t.study.reviewTab],
                  ['inbox', t.study.inboxTab]
                ] as const
              ).map(([tab, label]) => (
                <button
                  className={cn(
                    'rounded-lg px-4 py-2 text-sm font-medium transition-colors',
                    activeTab === tab ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
                  )}
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          {activeTab === 'today' && overview && (
            <StudyToday
              onOpenReview={() => setActiveTab('review')}
              onStartAgent={onStartAgentReview}
              overview={overview}
            />
          )}
          {activeTab === 'calendar' && (
            <>
              {selectedSchedule && scheduleValidation?.ok === false && (
                <div className="rounded-2xl border border-amber-500/40 bg-amber-500/10 p-5">
                  <h3 className="font-semibold">{t.study.invalidSchedule}</h3>
                  <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                    {scheduleValidation.errors.map(error => (
                      <li key={error}>{error}</li>
                    ))}
                  </ul>
                </div>
              )}
              {selectedSchedule && scheduleValidation?.ok === true && (
                <div className="space-y-5">
                  <div className="flex flex-wrap items-end justify-between gap-3">
                    <div>
                      <h3 className="text-2xl font-semibold">{selectedSchedule.title}</h3>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {t.study.range}: {selectedSchedule.range.start} → {selectedSchedule.range.end} ·{' '}
                        {t.study.timezone}: {selectedSchedule.timezone}
                      </p>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {selectedSchedule.phases.length} {t.study.phases} · {selectedSchedule.events.length}{' '}
                      {t.study.events}
                    </div>
                  </div>
                  {selectedSchedule.phases.length > 0 && (
                    <section className="rounded-3xl border bg-card/70 p-4">
                      <h4 className="text-lg font-semibold">{t.study.phases}</h4>
                      <div className="mt-4 grid gap-3 lg:grid-cols-2">
                        {selectedSchedule.phases.map(phase => (
                          <PhaseCard key={phase.id} phase={phase} />
                        ))}
                      </div>
                    </section>
                  )}
                  {selectedSchedule.events.length === 0 && (
                    <div className="rounded-2xl border border-dashed bg-card/50 p-5 text-sm text-muted-foreground">
                      {t.study.noConcreteSessions}
                    </div>
                  )}
                  {groupEvents(selectedSchedule.events).map(([month, weeks]) => (
                    <section className="rounded-3xl border bg-card/70 p-4" key={month}>
                      <h4 className="text-lg font-semibold">{month}</h4>
                      <div className="mt-4 space-y-4">
                        {weeks.map(([week, events]) => (
                          <div className="grid gap-3 md:grid-cols-[5rem_1fr]" key={`${month}-${week}`}>
                            <div className="pt-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                              {week}
                            </div>
                            <div className="space-y-3">
                              {events.map(event => (
                                <EventCard event={event} key={event.id} project={selectedProject} />
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  ))}
                </div>
              )}
            </>
          )}
          {activeTab === 'review' && (
            <Suspense fallback={<PageLoader label={t.study.loading} />}>
              <ReviewView client={client} onStartAgentReview={onStartAgentReview} />
            </Suspense>
          )}
          {activeTab === 'inbox' && overview && (
            <StudyInbox
              onDecide={(proposalId, action) =>
                selectedProjectId ? decideProposal(client, selectedProjectId, proposalId, action) : Promise.resolve()
              }
              onStartAgent={onStartAgentReview}
              overview={overview}
            />
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

function PhaseCard({ phase }: { phase: StudySchedulePhase }) {
  const { t } = useI18n()
  return (
    <article className="rounded-2xl border bg-background/70 p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-medium text-primary">
          {phase.start} → {phase.end}
        </div>
        {phase.effort_minutes != null && (
          <div className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            {formatEffort(phase.effort_minutes)}
          </div>
        )}
      </div>
      <h5 className="mt-2 text-base font-semibold">{phase.title}</h5>
      <p className="mt-2 text-sm text-muted-foreground">{phase.goal}</p>
      {(phase.status || phase.source_curricula?.length) && (
        <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
          {phase.status && (
            <div>
              <span className="text-muted-foreground">{t.study.status}: </span>
              {phase.status}
            </div>
          )}
          {phase.source_curricula?.length && (
            <div>
              <span className="text-muted-foreground">{t.study.source}: </span>
              {phase.source_curricula.join(', ')}
            </div>
          )}
        </div>
      )}
      {phase.goals?.length && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
          {phase.goals.map(goal => (
            <li key={goal}>{goal}</li>
          ))}
        </ul>
      )}
    </article>
  )
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-background/60 p-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 font-medium">{value}</div>
    </div>
  )
}

function EventCard({ event, project }: { event: StudyScheduleEvent; project: StudyProject | undefined }) {
  const { t } = useI18n()
  return (
    <article className="rounded-2xl border bg-background/70 p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h5 className="text-base font-semibold">{eventLabel(project, event)}</h5>
          <div className="mt-1 text-sm text-muted-foreground">
            {formatDateTime(event.start)} → {formatDateTime(event.end)}
          </div>
        </div>
        <div className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
          {event.duration_minutes}m
        </div>
      </div>
      <div className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
        <div>
          <span className="text-muted-foreground">{t.study.status}: </span>
          {event.status}
        </div>
        {event.source_curriculum && (
          <div>
            <span className="text-muted-foreground">{t.study.source}: </span>
            {event.source_curriculum}
          </div>
        )}
        <div>
          <span className="text-muted-foreground">{t.study.goals}: </span>
          {event.goals.join(', ')}
        </div>
      </div>
    </article>
  )
}
