/**
 * The end of the questionnaire. `handoffPrompt` is the first message of the first chat: the ask on
 * line one, then "About me" and "Before you start", every line from a picked option or a machine
 * fact. `finish` (Start) and `skipSetup` (Skip setup) leave the questionnaire for the default profile.
 */

import { $freeTierStatus, ackFreeTierNotice } from '@/store/free-tier'
import { acceptLocalSetupOffer, dismissLocalSetupOffer, noteHandoffSession } from '@/store/local-setup-offer'
import { clearFreeTierIntro } from '@/store/onboarding'
import {
  $activeGatewayProfile,
  ensureGatewayProfile,
  normalizeProfileKey,
  pinNewChatProfile,
  switchToDefaultProfile
} from '@/store/profile'
import type { SessionResumeResult } from '@/types/hermes'

import { type OnboardingRequester, setRun } from './due'
import { freeAccountState } from './facts'
import {
  type Answers,
  chosenName,
  chosenTask,
  connectorOptions,
  type Facts,
  FLOW,
  type FlowOption,
  type FlowStep,
  type LocalFit,
  pluginOptions,
  stepOptions,
  visibleSteps
} from './flow'
import { type InferenceClock, realClock, START_WAIT_MS, waitForInference } from './inference'
import { $questionnaire, closeQuestionnaire, type FirstChat, noteFirstChat, setPending, showPreparing } from './store'

export const NO_TASK_ASK = 'What can you help me with? Ask me what I want to do first.'

export const MEMORY_LINE = 'Save the About me lines to your memory of me.'

const OS_NAMES = new Map([
  ['darwin', 'macOS'],
  ['linux', 'Linux'],
  ['win32', 'Windows']
])

/** "Windows 11", "macOS 26": the OS family's name and its release. */
export function osLabel(family: null | string | undefined, release: null | string | undefined): null | string {
  return family ? [OS_NAMES.get(family) ?? family, release].filter(Boolean).join(' ') : null
}

/** "RTX Spark (NVIDIA N1X · Windows 11 · 128 GB RAM)", or `null` when the facts are missing. */
export function machineLine(facts: Facts): null | string {
  const summary = facts.machine

  if (!summary) {
    return null
  }

  const { machine } = summary

  const detail = [
    machine.cpu_model,
    osLabel(machine.os_family, machine.os_release),
    summary.has_nvidia_gpu && !summary.is_spark ? 'NVIDIA GPU' : null,
    machine.ram_gb ? `${machine.ram_gb} GB RAM` : null
  ].filter(Boolean)

  const label = summary.is_spark && machine.os_family === 'win32' ? 'RTX Spark' : summary.machine_kind

  return detail.length > 0 ? `${label} (${detail.join(' · ')})` : label
}

function pickedOptions(step: FlowStep, facts: Facts, answers: Answers): FlowOption[] {
  if (answers.skipped.includes(step.id)) {
    return []
  }

  const ids: readonly string[] = step.id === 'apps' ? answers.apps : step.id === 'connectors' ? answers.connectors : []

  return stepOptions(step, facts, answers).filter(option => ids.includes(option.id))
}

function appsLine(facts: Facts, answers: Answers): null | string {
  const offered = (options: FlowOption[], step: 'apps' | 'connectors') =>
    answers.skipped.includes(step) ? new Set<string>() : new Set(options.map(option => option.id))

  const plugins = offered(pluginOptions(facts), 'apps')
  const connectors = offered(connectorOptions(facts), 'connectors')
  const connectorRows = facts.connectors.status === 'ready' ? facts.connectors.rows : []

  const titles = [
    ...(facts.plugins ?? []).filter(row => plugins.has(row.name) && answers.apps.includes(row.name)).map(row => row.title),
    ...connectorRows.filter(row => connectors.has(row.id) && answers.connectors.includes(row.id)).map(row => row.label)
  ]

  return titles.length > 0 ? `Apps I use: ${titles.join(', ')}.` : null
}

function aboutLines(facts: Facts, answers: Answers): string[] {
  const name = chosenName(answers)
  const machine = machineLine(facts)

  return [
    name ? `Call me ${name}.` : null,
    machine ? `This machine: ${machine}.` : null,
    appsLine(facts, answers)
  ].filter((line): line is string => line !== null)
}

function beforeLines(facts: Facts, answers: Answers, flow: readonly FlowStep[]): string[] {
  const steps = visibleSteps(facts, answers, flow)
  const picked = steps.flatMap(step => pickedOptions(step, facts, answers))
  const task = chosenTask(facts, answers, flow)

  return [...picked, ...(task ? [task] : [])].flatMap(option => option.before ?? [])
}

const bullets = (lines: string[]) => lines.map(line => `- ${line}`)

export function handoffPrompt(facts: Facts, answers: Answers, flow: readonly FlowStep[] = FLOW): string {
  const ask = chosenTask(facts, answers, flow)?.ask ?? NO_TASK_ASK
  const about = aboutLines(facts, answers)
  const before = [...beforeLines(facts, answers, flow), ...(about.length > 0 ? [MEMORY_LINE] : [])]

  const sections = [
    [ask],
    about.length > 0 ? ['About me:', ...bullets(about)] : [],
    before.length > 0 ? ['Before you start:', ...bullets(before)] : []
  ].filter(section => section.length > 0)

  return sections.map(section => section.join('\n')).join('\n\n')
}

export interface HandoffDeps {
  /** The connected backend (the launch profile's). */
  request: OnboardingRequester
  launchProfile: string
  /** `session.create` in the default profile, then `prompt.submit`; answers the runtime session id. */
  openDefaultChat: (text: string, onCreated: (chat: FirstChat) => void) => Promise<string>
  /** Opens `chat` when its first message reached the backend; `false` when it did not. */
  openLandedChat: (chat: FirstChat) => Promise<boolean>
  /** Re-run the overlay's readiness round so it lands on the right screen after the questionnaire. */
  refreshReadiness: () => Promise<void>
  /** The built-in quick tour; resolves once it is closed. */
  runTour: () => Promise<void>
  /** Local-model quickstart for the default profile. */
  startQuickstart: (model: LocalFit) => Promise<void>
  clock?: InferenceClock
}

export type FinishResult = 'failed' | 'started'

function applyLocalAnswer(answers: Answers): void {
  if (answers.skipped.includes('local')) {
    return
  }

  if (answers.local === 'yes') {
    acceptLocalSetupOffer()
  } else if (answers.local === 'no') {
    dismissLocalSetupOffer()
  }
}

/** Start: wait for inference, settle the flags, close, tour, then the first chat in default. */
export async function finish(facts: Facts, answers: Answers, deps: HandoffDeps): Promise<FinishResult> {
  setPending('start')
  const ready = await waitForInference(deps.request, { clock: deps.clock })

  if (!ready.ok) {
    // D23: the prompt is dropped; confirmed accent and layout stay; the picker explains the failure.
    await setRun(deps.request, false)
    closeQuestionnaire('failed')
    await deps.refreshReadiness()

    return 'failed'
  }

  if (deps.launchProfile !== 'default') {
    // The default profile adopts the free account the launch profile's backend made.
    await deps.request('free_tier.provision', { profile: 'default' })
  }

  await setRun(deps.request, false, { markProfileOffered: true })
  applyLocalAnswer(answers)

  // D24: the free account was introduced here, so the ready screen must not follow.
  if (await ackFreeTierNotice(deps.request)) {
    clearFreeTierIntro()
  }

  await deps.refreshReadiness()
  closeQuestionnaire('done')

  // D19: after the overlay is gone and before the first message, so it never covers approval cards.
  if (answers.tour === 'quick' && !answers.skipped.includes('tour')) {
    await deps.runTour()
  }

  await openFirstChat(facts, answers, deps)

  return 'started'
}

/** The first chat in default, then the quickstart. Start's last step, and its retry once the questionnaire has closed. */
export async function openFirstChat(
  facts: Facts,
  answers: Answers,
  deps: Pick<HandoffDeps, 'openDefaultChat' | 'openLandedChat' | 'startQuickstart'>
): Promise<void> {
  // A failed try may have lost only the submit's reply; sending again would run the task twice.
  const earlier = $questionnaire.get().firstChat

  const sessionId =
    earlier && (await deps.openLandedChat(earlier))
      ? earlier.runtimeSessionId
      : await deps.openDefaultChat(handoffPrompt(facts, answers), noteFirstChat)

  noteHandoffSession(sessionId)

  // After session.create, so the first chat stays on the free tier while the model downloads.
  if (answers.local === 'yes' && !answers.skipped.includes('local') && facts.local) {
    void deps.startQuickstart(facts.local)
  }
}

/** The first message reached the backend: its turn is running or its user message is stored. */
export function firstMessageLanded(resumed: Pick<SessionResumeResult, 'messages' | 'running'>): boolean {
  return Boolean(resumed.running) || resumed.messages.some(message => message.role === 'user')
}

/** Aim the next new chat at the default profile (D17), whatever profile is active or was picked for a new chat. */
export async function targetDefaultProfile(): Promise<void> {
  if (normalizeProfileKey($activeGatewayProfile.get()) !== 'default') {
    switchToDefaultProfile()
    await ensureGatewayProfile('default')
  }

  // session.create ranks a new-chat pick (profile or agent route) above the active profile.
  pinNewChatProfile('default')
}

/** Resolves once the free account exists, has failed for good, or the Start wait has passed. */
function freeAccountSettled(clock: InferenceClock): Promise<void> {
  let stop = () => {}

  const settled = new Promise<void>(resolve => {
    stop = $freeTierStatus.listen(status => {
      if (freeAccountState(status) !== 'waiting') {
        resolve()
      }
    })
  })

  // A retryable failure (rate limit) can back off for minutes; Skip waits no longer than Start (D23).
  return Promise.race([settled, clock.sleep(START_WAIT_MS)]).finally(() => stop())
}

/**
 * Skip setup: done for good, the profile offer left alone (D13), the notice still owed (D24). While
 * the free account is still being made the overlay says "Starting Hermes…" and then moves on to the
 * ready screen or the picker on its own.
 */
export async function skipSetup(deps: Pick<HandoffDeps, 'clock' | 'refreshReadiness' | 'request'>): Promise<void> {
  setPending('skip')
  await setRun(deps.request, false)

  if (freeAccountState($freeTierStatus.get()) === 'waiting') {
    showPreparing()
    await freeAccountSettled(deps.clock ?? realClock)
  }

  closeQuestionnaire('skipped')
  await deps.refreshReadiness()
}
