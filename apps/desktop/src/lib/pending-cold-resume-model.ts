// A manual model pick made while a stored conversation has no live runtime
// only paints the composer: with a null session id the picker sends no
// config.set, and a historical conversation later continues through
// session.resume, which never consumes the pick — so the next prompt can run
// on the history's previous model (#138492). This slot remembers such a pick
// so the resume paths can apply it to the rebound runtime before the next
// prompt ships. The slot is a single entry keyed by the stored session id: a
// pick for another conversation never matches, and a newer pick replaces it.
import { type ComposerModelSource, setCurrentModel, setCurrentModelSource, setCurrentProvider } from '@/store/session'

export interface PendingColdResumeModelChoice {
  storedSessionId: string
  provider: string
  model: string
  prevModel: string
  prevProvider: string
  prevSource: ComposerModelSource
}

interface ModelSwitchResponse {
  confirm_required?: boolean
}

let pendingChoice: PendingColdResumeModelChoice | null = null

export const recordPendingColdResumeModelChoice = (choice: PendingColdResumeModelChoice): void => {
  pendingChoice = choice
}

export const takePendingColdResumeModelChoice = (storedSessionId: string): PendingColdResumeModelChoice | null => {
  if (!pendingChoice || pendingChoice.storedSessionId !== storedSessionId) {
    return null
  }

  const choice = pendingChoice
  pendingChoice = null

  return choice
}

export const _resetPendingColdResumeModelChoiceForTests = (): void => {
  pendingChoice = null
}

const rollbackComposer = (choice: PendingColdResumeModelChoice): void => {
  setCurrentModel(choice.prevModel)
  setCurrentProvider(choice.prevProvider)
  setCurrentModelSource(choice.prevSource)
}

// Applies a recorded pick to the rebound runtime. Value format and the moa
// session-only scope mirror the picker's own config.set so both paths land on
// the gateway identically. An expensive-model confirmation or a failed call
// rolls the composer back to the pre-pick route and lets the prompt through —
// never worse than the pre-fix behavior, which applied nothing.
export const applyPendingColdResumeModelChoice = async (
  storedSessionId: string,
  runtimeSessionId: string,
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
): Promise<boolean> => {
  const choice = takePendingColdResumeModelChoice(storedSessionId)

  if (!choice) {
    return false
  }

  const scope = choice.provider.toLowerCase() === 'moa' ? ' --session' : ''

  try {
    const result = await requestGateway<ModelSwitchResponse>('config.set', {
      session_id: runtimeSessionId,
      key: 'model',
      value: `${choice.model} --provider ${choice.provider}${scope}`
    })

    if (result?.confirm_required) {
      rollbackComposer(choice)

      return false
    }

    return true
  } catch {
    rollbackComposer(choice)

    return false
  }
}
