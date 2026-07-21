import { useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import type { ProjectInfo } from '@/types/hermes'

import { Panel, PanelHeader } from '../overlays/panel'

interface NattpassMissionInput {
  acceptanceCriteria: string[]
  goal: string
  project: ProjectInfo
}

interface NattpassLaunch extends NattpassMissionInput {
  prompt: string
}

interface NattpassetViewProps {
  activeProjectId: null | string
  onClose: () => void
  onLaunch: (mission: NattpassLaunch, isCurrent: () => boolean) => Promise<boolean> | boolean
  projects: ProjectInfo[]
}

export function nattpassProjectWorkspace(project: ProjectInfo): string {
  return (project.primary_path || project.folders.find(folder => folder.is_primary)?.path || '').trim()
}

export interface NattpassLaunchContext {
  activeProjectId: null | string
  cwd: string
  freshDraftReady: boolean
  onFreshDraftRoute: boolean
  profile: string
  project: null | ProjectInfo
  runtimeSessionId: null | string
  storedSessionId: null | string
}

export interface ExpectedNattpassLaunchContext {
  activeProjectId: null | string
  profile: string
  projectId: string
  workspace: string
}

export function nattpassLaunchContextMatches(
  expected: ExpectedNattpassLaunchContext,
  current: NattpassLaunchContext
): boolean {
  return Boolean(
    current.activeProjectId === expected.activeProjectId &&
    current.profile === expected.profile &&
    current.project?.id === expected.projectId &&
    !current.project.archived &&
    nattpassProjectWorkspace(current.project) === expected.workspace &&
    current.runtimeSessionId === null &&
    current.storedSessionId === null &&
    current.freshDraftReady &&
    current.cwd === expected.workspace &&
    current.onFreshDraftRoute
  )
}

export function buildNattpassMission({ acceptanceCriteria, goal, project }: NattpassMissionInput): string {
  const workspace = nattpassProjectWorkspace(project)
  const criteria = acceptanceCriteria.map((criterion, index) => `${index + 1}. ${criterion}`).join('\n')

  return `NATTPASSET — LIVE OPERATOR MISSION

Mission
${goal.trim()}

Acceptance criteria
${criteria}

Scope identity supplied by Hermes Desktop
- Project: ${project.name}
- Project id: ${project.id}
- Expected workspace: ${workspace}

Mandatory scope gate — do this before any work
Verify the actual cwd, repository identity, git remotes, branch, and worktree status against the scope above. If the identity is missing, ambiguous, stale, dirty in a way that makes the mission unsafe, or does not match, STOP before editing or launching subagents and ask for a decision under BESLUT KRÄVS. Never silently retarget the mission.

Execution contract
- Restate the mission and acceptance criteria in plain language, then create a concrete todo plan.
- Perform the work through Hermes' existing todo, delegation, Kanban, and cron capabilities as appropriate. Do not create or pretend there is a second agent system.
- Keep working until the acceptance criteria have been exercised with real evidence. Do not substitute mock or demo runtime data for execution.
- This launch is a live session, not durable overnight execution. If the mission must survive a runtime restart, say so and use an existing durable mechanism such as cron where it genuinely fits; otherwise list that limitation under KVAR.
- Surface blockers as soon as they prevent safe progress. Do not claim success for work that was not run or verified.

Final handoff
End with exactly these headings and put every claim under one of them:
KLART — what was actually delivered.
VERIFIERAT — concrete checks run and their observed results.
KVAR — incomplete work or limitations, including durability.
BESLUT KRÄVS — only decisions that require the operator; write “Inget” when none are needed.
`
}

export async function launchNattpassMission(
  mission: Pick<NattpassLaunch, 'project' | 'prompt'>,
  startFreshSessionDraft: (workspace: string) => void,
  waitForFreshDraftRoute: () => Promise<void>,
  submitPrompt: (prompt: string) => Promise<boolean> | boolean,
  isLaunchContextCurrent: () => boolean
): Promise<boolean> {
  const workspace = nattpassProjectWorkspace(mission.project)

  if (!workspace) {
    return false
  }

  startFreshSessionDraft(workspace)
  await waitForFreshDraftRoute()

  if (!isLaunchContextCurrent()) {
    return false
  }

  return submitPrompt(mission.prompt)
}

export function NattpassetView({ activeProjectId, onClose, onLaunch, projects }: NattpassetViewProps) {
  const { t } = useI18n()
  const copy = t.nattpasset

  const eligibleProjects = useMemo(
    () => projects.filter(project => !project.archived && Boolean(nattpassProjectWorkspace(project))),
    [projects]
  )

  const initialProject = eligibleProjects.some(project => project.id === activeProjectId)
    ? activeProjectId
    : eligibleProjects.length === 1
      ? eligibleProjects[0].id
      : null

  const [projectId, setProjectId] = useState<null | string>(initialProject)
  const [goal, setGoal] = useState('')
  const [criteriaText, setCriteriaText] = useState('')
  const [scopeConfirmed, setScopeConfirmed] = useState(false)
  const [launching, setLaunching] = useState(false)
  const launchGenerationRef = useRef(0)

  const selectedProject = eligibleProjects.find(project => project.id === projectId) ?? null

  const criteria = criteriaText
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)

  const canLaunch = Boolean(selectedProject && goal.trim() && criteria.length && scopeConfirmed && !launching)

  const invalidatePendingLaunch = () => {
    if (launching) {
      launchGenerationRef.current += 1
      setLaunching(false)
    }
  }

  const launch = async () => {
    if (!selectedProject || !canLaunch) {
      return
    }

    const requestId = launchGenerationRef.current + 1
    launchGenerationRef.current = requestId
    setLaunching(true)
    const mission = { acceptanceCriteria: criteria, goal: goal.trim(), project: selectedProject }
    let accepted = false

    try {
      accepted = await onLaunch(
        { ...mission, prompt: buildNattpassMission(mission) },
        () => launchGenerationRef.current === requestId
      )
    } catch {
      accepted = false
    }

    if (launchGenerationRef.current === requestId && !accepted) {
      setLaunching(false)
    }
  }

  return (
    <Panel closeLabel={copy.close} contentClassName="max-w-3xl self-center" onClose={onClose}>
      <PanelHeader subtitle={copy.subtitle} title={copy.title} />
      <div className="min-h-0 flex-1 overflow-y-auto px-1 pb-1">
        <p className="mb-5 text-sm text-(--ui-text-secondary)">{copy.intro}</p>

        <section className="mb-5">
          <h3 className="mb-1 text-xs font-semibold text-foreground">{copy.goalLabel}</h3>
          <Textarea
            aria-label={copy.goalQuestion}
            onChange={event => {
              invalidatePendingLaunch()
              setGoal(event.target.value)
            }}
            placeholder={copy.goalPlaceholder}
            value={goal}
          />
        </section>

        <section className="mb-5">
          <h3 className="mb-1 text-xs font-semibold text-foreground">{copy.criteriaLabel}</h3>
          <Textarea
            aria-label={copy.criteriaQuestion}
            onChange={event => {
              invalidatePendingLaunch()
              setCriteriaText(event.target.value)
            }}
            placeholder={copy.criteriaPlaceholder}
            value={criteriaText}
          />
          <p className="mt-1 text-xs text-(--ui-text-tertiary)">{copy.criteriaHint}</p>
        </section>

        <section className="mb-5">
          <h3 className="mb-1 text-xs font-semibold text-foreground">{copy.scopeLabel}</h3>
          {eligibleProjects.length ? (
            <div className="space-y-1">
              {eligibleProjects.map(project => {
                const workspace = nattpassProjectWorkspace(project)
                const selected = project.id === projectId

                return (
                  <button
                    aria-pressed={selected}
                    className="row-hover flex w-full cursor-pointer items-start gap-2 rounded-md px-2 py-2 text-left"
                    key={project.id}
                    onClick={() => {
                      invalidatePendingLaunch()
                      setProjectId(project.id)
                      setScopeConfirmed(false)
                    }}
                    type="button"
                  >
                    <span aria-hidden="true" className="mt-1 text-(--ui-text-tertiary)">{selected ? '●' : '○'}</span>
                    <span className="min-w-0">
                      <span className="block text-sm font-medium text-foreground">{project.name}</span>
                      <span className="block truncate text-xs text-(--ui-text-tertiary)">{workspace}</span>
                    </span>
                  </button>
                )
              })}
            </div>
          ) : (
            <p className="text-sm text-(--ui-text-secondary)">{copy.noProject}</p>
          )}

          {selectedProject ? (
            <label className="mt-3 flex cursor-pointer items-start gap-2 text-sm text-(--ui-text-secondary)">
              <input
                checked={scopeConfirmed}
                className="mt-0.5"
                onChange={event => {
                  invalidatePendingLaunch()
                  setScopeConfirmed(event.target.checked)
                }}
                type="checkbox"
              />
              <span>{copy.confirmScope(selectedProject.name, nattpassProjectWorkspace(selectedProject))}</span>
            </label>
          ) : null}
        </section>

        <p className="mb-5 text-xs text-(--ui-text-tertiary)">{copy.liveNotice}</p>

        <div className="flex justify-end gap-2">
          <Button onClick={onClose} variant="secondary">{copy.cancel}</Button>
          <Button disabled={!canLaunch} onClick={() => void launch()}>
            {launching ? copy.launching : copy.launch}
          </Button>
        </div>
      </div>
    </Panel>
  )
}
