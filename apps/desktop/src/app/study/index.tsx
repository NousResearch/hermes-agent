import { useStore } from '@nanostores/react'
import { lazy, Suspense, useCallback, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getStudyProjects, getStudySchedule, getStudySchedules } from '@/hermes'
import { useI18n } from '@/i18n'
import { validateStudySchedule } from '@/lib/study-schemas'
import { cn } from '@/lib/utils'
import {
  $studyConfigured,
  $studyError,
  $studyLoadState,
  $studyMessage,
  $studyProjects,
  $studySchedules,
  $studySelectedProjectId,
  $studySelectedSchedule,
  $studySelectedScheduleId
} from '@/store/study'
import type { StudyProject, StudyScheduleEvent, StudyScheduleSummary } from '@/types/hermes'

import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

const ReviewView = lazy(async () => ({ default: (await import('./review')).ReviewView }))

interface StudyViewProps {
  onClose?: () => void
  onStartAgentReview?: (prompt: string) => void | Promise<void>
}

function formatDateTime(value: string): string {
  return value.slice(0, 16).replace('T', ' ')
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
  return project?.subjects.find(subject => subject.id === subjectId)?.label ?? subjectId
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
      <div className="mt-1 text-xs">
        {project.exam_type} · {project.exam_date}
      </div>
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
        {schedule.range.start} → {schedule.range.end} · {schedule.event_count}
      </div>
    </button>
  )
}

export function StudyView({ onClose, onStartAgentReview }: StudyViewProps) {
  const { t } = useI18n()
  const projects = useStore($studyProjects)
  const schedules = useStore($studySchedules)
  const selectedProjectId = useStore($studySelectedProjectId)
  const selectedScheduleId = useStore($studySelectedScheduleId)
  const selectedSchedule = useStore($studySelectedSchedule)
  const loadState = useStore($studyLoadState)
  const error = useStore($studyError)
  const configured = useStore($studyConfigured)
  const message = useStore($studyMessage)
  const selectedProject = projects.find(project => project.project_id === selectedProjectId)
  const scheduleValidation = selectedSchedule ? validateStudySchedule(selectedSchedule, selectedProject) : null
  const [activeTab, setActiveTab] = useState<'calendar' | 'review'>('calendar')

  const loadProjects = useCallback(async () => {
    $studyLoadState.set('loading')
    $studyError.set(null)
    try {
      const response = await getStudyProjects()
      $studyConfigured.set(response.configured)
      $studyMessage.set(response.message ?? null)
      $studyProjects.set(response.projects)
      const nextProjectId = response.projects[0]?.project_id ?? null
      $studySelectedProjectId.set(nextProjectId)
      if (!response.configured || !nextProjectId) {
        $studySchedules.set([])
        $studySelectedScheduleId.set(null)
        $studySelectedSchedule.set(null)
        $studyLoadState.set('ready')
        return
      }
      const scheduleResponse = await getStudySchedules(nextProjectId)
      $studySchedules.set(scheduleResponse.schedules)
      const nextScheduleId = scheduleResponse.schedules[0]?.schedule_id ?? null
      $studySelectedScheduleId.set(nextScheduleId)
      $studySelectedSchedule.set(nextScheduleId ? await getStudySchedule(nextProjectId, nextScheduleId) : null)
      $studyLoadState.set('ready')
    } catch (err) {
      $studyError.set(err instanceof Error ? err.message : String(err))
      $studyLoadState.set('error')
    }
  }, [])

  const selectProject = useCallback(async (projectId: string) => {
    $studyLoadState.set('loading')
    $studyError.set(null)
    try {
      $studySelectedProjectId.set(projectId)
      const scheduleResponse = await getStudySchedules(projectId)
      $studySchedules.set(scheduleResponse.schedules)
      const nextScheduleId = scheduleResponse.schedules[0]?.schedule_id ?? null
      $studySelectedScheduleId.set(nextScheduleId)
      $studySelectedSchedule.set(nextScheduleId ? await getStudySchedule(projectId, nextScheduleId) : null)
      $studyLoadState.set('ready')
    } catch (err) {
      $studyError.set(err instanceof Error ? err.message : String(err))
      $studyLoadState.set('error')
    }
  }, [])

  const selectSchedule = useCallback(
    async (scheduleId: string) => {
      if (!selectedProjectId) {
        return
      }
      $studyLoadState.set('loading')
      $studyError.set(null)
      try {
        $studySelectedScheduleId.set(scheduleId)
        $studySelectedSchedule.set(await getStudySchedule(selectedProjectId, scheduleId))
        $studyLoadState.set('ready')
      } catch (err) {
        $studyError.set(err instanceof Error ? err.message : String(err))
        $studyLoadState.set('error')
      }
    },
    [selectedProjectId]
  )

  useEffect(() => {
    void loadProjects()
  }, [loadProjects])

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
                  onSelect={() => void selectProject(project.project_id)}
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
                  onSelect={() => void selectSchedule(schedule.schedule_id)}
                  schedule={schedule}
                />
              ))}
              {selectedProjectId && schedules.length === 0 && (
                <div className="text-sm text-muted-foreground">{t.study.noSchedules}</div>
              )}
            </div>
          </div>
        </OverlaySidebar>
        <OverlayMain className="overflow-y-auto p-6 pt-[calc(var(--titlebar-height)+1.5rem)]">
          {loadState === 'loading' && !selectedSchedule && <PageLoader label={t.study.loading} />}
          {!configured && (
            <div className="rounded-2xl border border-dashed bg-card/70 p-8">
              <h2 className="text-xl font-semibold">{message || t.study.notConfigured}</h2>
              <p className="mt-2 text-sm text-muted-foreground">{t.study.subtitle}</p>
            </div>
          )}
          {error && (
            <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-5 text-sm">
              <div className="font-semibold">{error}</div>
              <Button className="mt-4" onClick={() => void loadProjects()} size="sm" variant="secondary">
                {t.study.retry}
              </Button>
            </div>
          )}
          {selectedProject && (
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
                <Meta label={t.study.examDate} value={selectedProject.exam_date} />
                <Meta label={t.study.phase} value={selectedProject.phase} />
                <Meta label={t.study.timezone} value={selectedProject.timezone} />
              </div>
            </div>
          )}
          <div className="mb-4 flex gap-1">
            <button
              className={cn(
                'rounded-lg px-4 py-2 text-sm font-medium transition-colors',
                activeTab === 'calendar' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
              onClick={() => setActiveTab('calendar')}
              type="button"
            >
              {t.study.calendarTab}
            </button>
            <button
              className={cn(
                'rounded-lg px-4 py-2 text-sm font-medium transition-colors',
                activeTab === 'review' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
              onClick={() => setActiveTab('review')}
              type="button"
            >
              {t.study.reviewTab}
            </button>
          </div>
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
                      {selectedSchedule.events.length} {t.study.events}
                    </div>
                  </div>
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
              <ReviewView onStartAgentReview={onStartAgentReview} />
            </Suspense>
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
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
