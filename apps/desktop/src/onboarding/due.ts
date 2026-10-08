/**
 * Whether the questionnaire runs, and the one flag that says so (`onboarding.run` in the root
 * profile's config). `onboarding.state` answers `{run, eligible}`; an older backend answers another
 * shape under the same name or lacks the method, and both read as "not due" (version skew).
 */

import type { OnboardingRunStateResult } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { withTimeout } from '@/lib/with-timeout'
import type { FreeTierRequester } from '@/store/free-tier'
import { markQuestionnaireDecided, QUESTIONNAIRE_DECIDE_DEADLINE_MS } from '@/store/onboarding-presence'
import { $connection } from '@/store/session'

import { openQuestionnaire } from './store'

/**
 * D2: a free-tier build (the launch flag the backend spawn also carries) on the local primary
 * backend; a remote host is not this computer.
 */
export const mayRunHere = () => window.hermesDesktop?.guestOnboardingEnabled === true && $connection.get()?.mode === 'local'

/**
 * A local backend's due check answered eligible. Every local backend (the primary and each pooled
 * profile's, on its own port) reads the same root-profile answer, so this holds for all of them.
 */
const $eligibleLocally = atom(false)

/** The questionnaire can run on the current connection: Settings shows Run setup again. */
export const $questionnaireAvailable = computed(
  [$eligibleLocally, $connection],
  (eligible, connection) => eligible && connection?.mode === 'local'
)

/** The gateway requester the free-tier store already takes; the questionnaire shares its reads. */
export type OnboardingRequester = FreeTierRequester

const isBool = (value: boolean | null | string | undefined): value is boolean => value === true || value === false

/**
 * The new `{run, eligible}` shape, or `null`. An older backend answers `{eligible, intro, …}` under
 * the same name (no `run`), or something else entirely; neither is due.
 */
type RunStateAnswer = null | Partial<Record<keyof OnboardingRunStateResult, boolean | null | string>> | undefined

export function readRunState(value: RunStateAnswer): null | OnboardingRunStateResult {
  const run = value?.run
  const eligible = value?.eligible

  return isBool(run) && isBool(eligible) ? { eligible, run } : null
}

async function readDue(request: OnboardingRequester): Promise<null | OnboardingRunStateResult> {
  try {
    return readRunState(
      await withTimeout(request<RunStateAnswer>('onboarding.state'), QUESTIONNAIRE_DECIDE_DEADLINE_MS, 'onboarding.state timed out')
    )
  } catch {
    return null
  }
}

/** Due when the backend is the new shape and says both. Any failure is "not due". */
export async function questionnaireDue(request: OnboardingRequester): Promise<boolean> {
  const state = await readDue(request)

  return Boolean(state?.eligible && state.run)
}

/** Read the due check once per launch, open when due, and release everything waiting on the answer. */
export async function decideQuestionnaire(request: OnboardingRequester): Promise<boolean> {
  const state = await readDue(request)
  const due = Boolean(state?.eligible && state.run)

  $eligibleLocally.set(state?.eligible === true)

  if (due) {
    openQuestionnaire()
  }

  markQuestionnaireDecided()

  return due
}

/** Start and Skip write `false` (Start also marks the profile offer as made, D13); Run setup again writes `true`. */
export async function setRun(
  request: OnboardingRequester,
  run: boolean,
  { markProfileOffered = false }: { markProfileOffered?: boolean } = {}
): Promise<void> {
  await request('onboarding.set_run', { mark_profile_offered: markProfileOffered, run })
}

/** Settings → Run setup again: due on the next launch too, and open now without a reload. */
export async function runSetupAgain(request: OnboardingRequester): Promise<void> {
  // The Settings row hides as the connection moves; this covers a click made on the way out.
  if (!$questionnaireAvailable.get()) {
    return
  }

  await setRun(request, true)
  openQuestionnaire()
}
