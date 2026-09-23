/**
 * Carries the guide's install card result to the handoff (NS-960 D6). The guide calls manage_catalog once,
 * as the last beat before handoff; when that card settles, each plugin row's outcome is written into the
 * onboarding answers so the build session's runbook can say what is ready and what was offered and not
 * installed. The build agent has no install tool, so this record is the only way it learns either.
 */
import type { ConnectionRequest, ConnectionTarget } from '@/store/connection-request'
import { $connectionRequests } from '@/store/connection-request'
import { $onboardingAnswers, type PluginOutcome, setOnboardingAnswers } from '@/store/onboarding-answers'

const OUTCOME = { connected: 'installed', failed: 'failed' } as const satisfies Partial<
  Record<ConnectionTarget['state'], PluginOutcome['state']>
>

const outcomeState = (state: ConnectionTarget['state']): PluginOutcome['state'] =>
  state === 'connected' || state === 'failed' ? OUTCOME[state] : 'skipped'

/** Plugin rows of a settled card, keyed by catalog name. Null while the card is open or has no plugin row. */
export function pluginOutcomesFrom(request: ConnectionRequest): null | Record<string, PluginOutcome> {
  if (!request.settled) {
    return null
  }

  const rows = request.targets.filter(target => target.kind === 'plugin')

  if (rows.length === 0) {
    return null
  }

  return Object.fromEntries(
    rows.map(target => [target.name, { detail: target.detail, state: outcomeState(target.state), tools: target.tools }])
  )
}

/** Record outcomes from the guide session's cards. Returns the unsubscribe. */
export function watchPluginOutcomes(guideRuntimeId: () => null | string | undefined): () => void {
  return $connectionRequests.subscribe(requests => {
    const id = guideRuntimeId()
    const request = id ? requests[id] : undefined
    const outcomes = request ? pluginOutcomesFrom(request) : null

    if (!outcomes) {
      return
    }

    const current = $onboardingAnswers.get().pluginOutcomes

    if (JSON.stringify({ ...current, ...outcomes }) !== JSON.stringify(current)) {
      setOnboardingAnswers({ pluginOutcomes: { ...current, ...outcomes } })
    }
  })
}
