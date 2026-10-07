import type { ConnectionRequest, ConnectionTarget } from '@/store/connection-request'
import { $connectionRequests } from '@/store/connection-request'
import { $onboardingAnswers, type PluginOutcome, setOnboardingAnswers } from '@/store/onboarding-answers'

const OUTCOME = { connected: 'installed', failed: 'failed' } as const satisfies Partial<
  Record<ConnectionTarget['state'], PluginOutcome['state']>
>

const outcomeState = (state: ConnectionTarget['state']): PluginOutcome['state'] =>
  state === 'connected' || state === 'failed' ? OUTCOME[state] : 'skipped'

export function pluginOutcomesFrom(request: ConnectionRequest): null | Record<string, PluginOutcome> {
  if (!request.settled) {
    return null
  }

  const rows = request.targets.filter(target => target.kind === 'plugin')

  if (rows.length === 0) {
    return null
  }

  return Object.fromEntries(
    rows.map(target => [
      target.name,
      {
        detail: target.detail,
        skill: target.catalog?.skill ?? '',
        state: outcomeState(target.state),
        tools: target.tools
      }
    ])
  )
}

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
