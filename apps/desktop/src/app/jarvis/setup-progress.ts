/**
 * "Start z Czeskiem": the handful of things that turn a fresh install into a useful one, and which of
 * them are already true. Pure on purpose — the card feeds it real state (model, connections, vault,
 * browser login, history) and this only decides what counts as done and what is next.
 */

export const SETUP_STEP_IDS = ['model', 'google', 'memory', 'browser', 'firstTask', 'briefing'] as const

export type SetupStepId = (typeof SETUP_STEP_IDS)[number]

export interface SetupInput {
  /** The user has already asked for a briefing from this card. */
  briefingTried: boolean
  /** Signed in to Google in the agent's own browser or in the copied logins. */
  browserSignedIn?: boolean
  googleConnected?: boolean
  hasHistory: boolean
  modelReady: boolean
  vaultExists?: boolean
}

export interface SetupStep {
  done: boolean
  id: SetupStepId
}

export interface SetupChecklist {
  complete: boolean
  doneCount: number
  /** First step still to do — the one the card highlights. */
  next: SetupStepId | null
  steps: SetupStep[]
  total: number
}

export function buildSetupChecklist(input: SetupInput): SetupChecklist {
  const done: Record<SetupStepId, boolean> = {
    briefing: input.briefingTried,
    browser: input.browserSignedIn === true,
    firstTask: input.hasHistory,
    google: input.googleConnected === true,
    memory: input.vaultExists === true,
    model: input.modelReady
  }

  const steps = SETUP_STEP_IDS.map(id => ({ done: done[id], id }))
  const doneCount = steps.filter(step => step.done).length

  return {
    complete: doneCount === steps.length,
    doneCount,
    next: steps.find(step => !step.done)?.id ?? null,
    steps,
    total: steps.length
  }
}
