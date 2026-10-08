/**
 * What the flow reads, fetched in parallel when the questionnaire opens. Each source fails on its own:
 * a failed read only hides or plainifies its own step. The connector list needs the free account
 * (the list is scoped to its identity), so it stays `loading` until `setup.ready` and is read again then.
 */

import type { ConnectorsListResult, MachineFactsResult, PluginsManageResult } from '@hermes/shared'

import { listConnectors } from '@/app/capabilities/connectors/data/rpc'
import { connectorTitle } from '@/lib/connector-tools'
import { $freeTierStatus, freeTierSetupFailure, refreshFreeTierStatus } from '@/store/free-tier'
import { $setupReadyTick } from '@/store/live-sync'
import { readLocalSetupEligibility } from '@/store/local-setup-offer'
import type { FreeTierStatus } from '@/types/hermes'

import type { OnboardingRequester } from './due'
import { type ConnectorList, type Facts, type LocalFit, PLUGIN_APP_NAMES } from './flow'
import { orderConnectorPicks } from './visuals/options'

export interface FactSources {
  request: OnboardingRequester
  /** `connectors.list` for the account owner in the default profile (D17). */
  listConnectors: () => Promise<ConnectorsListResult>
  localFit: () => Promise<LocalFit | null>
}

export const defaultFactSources = (request: OnboardingRequester): FactSources => ({
  listConnectors: () => listConnectors('default'),
  localFit: () =>
    readLocalSetupEligibility().then(({ fit }) => (fit ? { id: fit.model.id, name: fit.model.display_name } : null)),
  request
})

/** Whether the account the connector list needs exists, may still be coming (or is a signed-in one), or will not come. */
export function freeAccountState(status: FreeTierStatus | null): 'failed' | 'ready' | 'waiting' {
  if (!status?.enabled || status.has_guest) {
    // No free tier here (an account of the user's own, or an older backend): the list answers for itself.
    return 'ready'
  }

  const failure = freeTierSetupFailure(status)

  return failure && !failure.retryable ? 'failed' : 'waiting'
}

export async function readConnectorList(sources: FactSources): Promise<ConnectorList> {
  try {
    const result = await sources.listConnectors()
    const rows = orderConnectorPicks(result.connectors).map(row => ({ id: row.connector, label: connectorTitle(row.connector) }))

    return result.available && rows.length > 0 ? { rows, status: 'ready' } : { status: 'unavailable' }
  } catch {
    return { status: 'unavailable' }
  }
}

async function readMachine(sources: FactSources): Promise<MachineFactsResult | null> {
  return sources.request<MachineFactsResult>('machine.facts').catch(() => null)
}

async function readPlugins(sources: FactSources): Promise<Facts['plugins']> {
  return sources
    .request<PluginsManageResult>('plugins.manage', { action: 'presence', names: PLUGIN_APP_NAMES })
    .then(result => result.presence ?? [])
    .catch(() => [])
}

/** Read every fact, report each as it lands, and re-read connectors on `setup.ready`. Returns the stop. */
export function watchFacts(sources: FactSources, report: (facts: Partial<Facts>) => void): () => void {
  let stopped = false
  let generation = 0

  const reportLive = (facts: Partial<Facts>) => {
    if (!stopped) {
      report(facts)
    }
  }

  const readConnectors = (status: FreeTierStatus | null) => {
    const account = freeAccountState(status)
    const mine = ++generation

    if (account === 'failed') {
      reportLive({ connectors: { status: 'unavailable' } })

      return
    }

    // `waiting` also covers a signed-in Nous account, which never gets a free one: its list answers now.
    // A pending free account's list is not ready yet, so it stays `loading` until `setup.ready`.
    void readConnectorList(sources).then(
      connectors =>
        mine === generation &&
        reportLive({ connectors: account === 'waiting' && connectors.status !== 'ready' ? { status: 'loading' } : connectors })
    )
  }

  void readMachine(sources).then(machine => reportLive({ machine }))
  void readPlugins(sources).then(plugins => reportLive({ plugins }))
  void sources
    .localFit()
    .catch(() => null)
    .then(local => reportLive({ local }))
  void refreshFreeTierStatus(sources.request).then(readConnectors)

  const stopReady = $setupReadyTick.listen(() => void refreshFreeTierStatus(sources.request).then(readConnectors))
  const stopStatus = $freeTierStatus.listen(status => freeAccountState(status) === 'failed' && readConnectors(status))

  return () => {
    stopped = true
    stopReady()
    stopStatus()
  }
}
