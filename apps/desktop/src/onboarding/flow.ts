/**
 * The whole first-run questionnaire as data. Change the onboarding by editing this file.
 *
 * Steps show by predicates over the machine facts and earlier answers. Options carry the
 * prompt lines they add to the first chat (`about`, `before`, `ask`), so the handoff prompt
 * has no separate template. Labels are i18n lookups; prompt lines are English because the
 * model reads them, not the person.
 */

import type { CatalogPluginPresence, MachineFactsResult } from '@hermes/shared'

import type { QuestionnaireTranslations } from '@/i18n/types_questionnaire'

export type StepId = 'accent' | 'apps' | 'connectors' | 'layout' | 'local' | 'name' | 'task' | 'tour'

export type Label = (copy: QuestionnaireTranslations) => string

export interface LocalFit {
  /** Catalog id handed to quickstart. */
  id: string
  name: string
}

export interface ConnectorPick {
  id: string
  label: string
}

/** `loading` until the free account exists (`setup.ready`); `unavailable` hides the step. */
export type ConnectorList =
  | { status: 'loading' }
  | { rows: ConnectorPick[]; status: 'ready' }
  | { status: 'unavailable' }

export interface Facts {
  /** `null` when `machine.facts` failed: no machine line, no Spark copy. */
  machine: MachineFactsResult | null
  /** Presence rows for `PLUGIN_APPS`, in that order; empty when the read failed, `undefined` while it is out. */
  plugins: CatalogPluginPresence[] | undefined
  /** The local model this machine fits (`readLocalSetupEligibility`), else `null`; `undefined` while it is read. */
  local: LocalFit | null | undefined
  connectors: ConnectorList
}

export interface TaskAnswer {
  id: string
  /** Free text for `other`. */
  text?: string
}

export interface Answers {
  name?: string
  /** Stored accent swatch id; `null` is the theme's own accent (Nous blue). */
  accent?: null | string
  layout?: string
  local?: 'no' | 'yes'
  apps: string[]
  connectors: string[]
  task?: TaskAnswer
  tour?: 'none' | 'quick'
  skipped: StepId[]
}

export const EMPTY_ANSWERS: Answers = { apps: [], connectors: [], skipped: [] }

export interface PromptLines {
  about?: string[]
  before?: string[]
  ask?: string
}

export interface FlowOption extends PromptLines {
  id: string
  label: Label
  detail?: Label
  /** Plugin rows: the catalog's "Disclosure:" sentence, shown when picked. */
  disclosure?: string
}

export interface FlowStep {
  id: StepId
  when?: (facts: Facts, answers: Answers) => boolean
  options?: (facts: Facts, answers: Answers) => FlowOption[]
  multi?: boolean
  /** A free-text "Other" pill: name and task only. */
  other?: boolean
}

/** Plugin apps the questionnaire may offer, by catalog name. Offered only when installed (D7). */
export const PLUGIN_APPS: ReadonlyArray<{ name: string; before?: string[] }> = [
  {
    name: 'blender',
    before: [
      'If the Blender plugin will not connect, write ~/hermes-first-task/first_scene.py and give me the blender --python command.'
    ]
  },
  { name: 'nvidia-app' },
  { name: 'nvidia-broadcast' }
]

export const PLUGIN_APP_NAMES = PLUGIN_APPS.map(app => app.name)

const OFFERABLE = new Set(['app_not_running', 'present'])

const TASK_LINES = ['Put new files in ~/hermes-first-task/.', 'Do one small piece first, show me, then ask how it looks.']

/** Connectors a daily brief can read from. */
const BRIEF_FEEDS = new Set(['github', 'gmail', 'googlecalendar', 'linear', 'notion', 'slack'])

const MAX_TASKS = 4

export function firstName(fullName: null | string | undefined): null | string {
  return fullName?.trim().split(/\s+/)[0] || null
}

function nameOptions(facts: Facts): FlowOption[] {
  const suggested = firstName(facts.machine?.full_name)

  return suggested ? [{ about: [`Call me ${suggested}.`], id: suggested, label: () => suggested }] : []
}

export function pluginOptions(facts: Facts): FlowOption[] {
  return PLUGIN_APPS.flatMap(app => {
    const row = facts.plugins?.find(candidate => candidate.name === app.name)

    if (!row || !OFFERABLE.has(row.state)) {
      return []
    }

    return [
      {
        before: [`Install the ${row.title} plugin (catalog name: ${row.name}).`, ...(app.before ?? [])],
        detail: copy => (row.state === 'app_not_running' ? copy.apps.pluginNotRunning : copy.apps.plugin),
        disclosure: row.disclosure,
        id: row.name,
        label: () => row.title
      }
    ]
  })
}

export function connectorOptions(facts: Facts): FlowOption[] {
  if (facts.connectors.status !== 'ready') {
    return []
  }

  return facts.connectors.rows.map(row => ({
    before: [`Connect ${row.label} (connector: ${row.id}).`],
    id: row.id,
    label: () => row.label
  }))
}

export function pickedLabels(options: FlowOption[], ids: readonly string[], copy: QuestionnaireTranslations): string[] {
  return options.filter(option => ids.includes(option.id)).map(option => option.label(copy))
}

// D20: no machine-setup task; the local-model step covers that.
function taskOptions(facts: Facts, answers: Answers): FlowOption[] {
  const task = (id: string, label: Label, ask: string): FlowOption => ({ ask, before: TASK_LINES, id, label })
  const rows = facts.connectors.status === 'ready' ? facts.connectors.rows : []
  const feeds = rows.filter(row => BRIEF_FEEDS.has(row.id) && answers.connectors.includes(row.id))

  const candidates: Array<false | FlowOption> = [
    answers.apps.includes('blender') &&
      task('blender', c => c.task.blender, 'Build me a small scene in Blender: a desk, a lamp and a mug, lit for a render.'),
    answers.apps.includes('nvidia-app') &&
      task('nvidia', c => c.task.nvidia, 'Check my NVIDIA driver and tune graphics settings for the game I play most.'),
    answers.apps.includes('nvidia-broadcast') &&
      task(
        'broadcast',
        c => c.task.broadcast,
        'Set up NVIDIA Broadcast for calls: noise removal on my mic and a background blur.'
      ),
    feeds.length > 0 &&
      task(
        'brief',
        c => c.task.brief,
        `Give me a short brief of today from ${feeds.map(feed => feed.label).join(', ')}.`
      ),
    task('tidy', c => c.task.tidy, 'Tidy my Downloads folder. Show me a dry-run plan before you move anything.')
  ]

  return candidates.filter((option): option is FlowOption => option !== false).slice(0, MAX_TASKS)
}

const TOUR_OPTIONS: FlowOption[] = [
  { detail: c => c.tour.quickDetail, id: 'quick', label: c => c.tour.quick },
  { detail: c => c.tour.noneDetail, id: 'none', label: c => c.tour.none }
]

function localOptions(facts: Facts): FlowOption[] {
  if (!facts.local) {
    return []
  }

  const model = facts.local.name

  return [
    { detail: c => c.local.downloadDetail, id: 'yes', label: c => c.local.download(model) },
    { detail: c => c.local.notNowDetail, id: 'no', label: c => c.local.notNow }
  ]
}

export const STEPS = {
  name: { id: 'name', options: nameOptions, other: true },
  accent: { id: 'accent' },
  layout: { id: 'layout' },
  // A step whose facts are still being read stays in the flow, so it is not passed before it can be asked.
  local: { id: 'local', options: localOptions, when: facts => facts.local !== null },
  apps: {
    id: 'apps',
    multi: true,
    options: pluginOptions,
    when: facts => facts.plugins === undefined || pluginOptions(facts).length > 0
  },
  connectors: {
    id: 'connectors',
    multi: true,
    options: connectorOptions,
    when: facts => facts.connectors.status !== 'unavailable'
  },
  task: { id: 'task', options: taskOptions, other: true },
  tour: { id: 'tour', options: () => TOUR_OPTIONS }
} satisfies Record<StepId, FlowStep>

/** The order the questions come in. */
export const FLOW: readonly FlowStep[] = [
  STEPS.name,
  STEPS.accent,
  STEPS.layout,
  STEPS.local,
  STEPS.apps,
  STEPS.connectors,
  STEPS.task,
  STEPS.tour
]

export function visibleSteps(facts: Facts, answers: Answers, flow: readonly FlowStep[] = FLOW): FlowStep[] {
  return flow.filter(step => !step.when || step.when(facts, answers))
}

export function stepOptions(step: FlowStep, facts: Facts, answers: Answers): FlowOption[] {
  return step.options?.(facts, answers) ?? []
}

/** The task as the prompt sees it: a picked option that is still offered, or the typed text. */
export function chosenTask(facts: Facts, answers: Answers, flow: readonly FlowStep[] = FLOW): FlowOption | null {
  const task = answers.task
  const step = flow.find(candidate => candidate.id === 'task')

  if (!task || !step || answers.skipped.includes('task')) {
    return null
  }

  if (task.id === 'other') {
    const text = task.text?.trim()

    return text ? { ask: text, before: TASK_LINES, id: 'other', label: () => text } : null
  }

  return stepOptions(step, facts, answers).find(option => option.id === task.id) ?? null
}

/** The name the prompt uses, or `null` when the step was skipped or left empty. */
export function chosenName(answers: Answers): null | string {
  return answers.skipped.includes('name') ? null : answers.name?.trim() || null
}
