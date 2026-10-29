/**
 * Local-setup eligibility: whether this machine can run a model locally, read
 * once per backend connection from the local-models status + catalog and from
 * the configured providers' endpoints.
 *
 * Two surfaces read it: the "Run locally" row at the top of the model menu
 * (until setup is done) and the handoff tour's model-pill step. A machine with
 * a runtime and a staged model no longer qualifies, so both hide — and neither
 * does a machine whose provider already points at a local endpoint, where the
 * user runs their own server.
 */

import { atom, computed } from 'nanostores'

import { getGlobalModelOptions, getLocalCatalog, getLocalModelsStatus } from '@/hermes'
import { isLoopbackEndpoint } from '@/lib/local-endpoint'
import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

import { $localModelsEnabled } from './local-models-flag'
import { $connection } from './session'

/** The recommended catalog row that fits, when this machine qualifies. */
interface LocalSetupFit {
  model: LocalCatalogModel
}

interface Eligibility {
  checkedAt: number
  fit: LocalSetupFit | null
  reason: string
  /** A failed read or a connection not up yet: shown in `state()` but never cached, so the next read retries. */
  transient?: boolean
}

/** Session cache: eligibility is a fact about the backend's machine, so a connection change drops it. */
export const $localSetupEligibility = atom<Eligibility | null>(null)

let eligibilityRead: Promise<Eligibility> | null = null
/** Bumped on every invalidation: a read that started before it must not land after it. */
let eligibilityGeneration = 0

function pickLocalSetupFit(
  connectionMode: null | string,
  status: LocalModelsStatus | null,
  catalog: readonly LocalCatalogModel[] | null,
  localEndpointInUse: boolean
): Eligibility {
  const at = Date.now()

  if (connectionMode === null) {
    return { checkedAt: at, fit: null, reason: 'connection not established yet', transient: true }
  }

  if (connectionMode !== 'local') {
    return { checkedAt: at, fit: null, reason: `connection is ${connectionMode}, not local` }
  }

  // A provider already pointed at this machine runs a model the user chose: the offer would ask
  // them to replace it.
  if (localEndpointInUse) {
    return { checkedAt: at, fit: null, reason: 'a provider already points at a local endpoint' }
  }

  if (!status || !catalog) {
    return { checkedAt: at, fit: null, reason: 'no local-models status or catalog' }
  }

  if (status.runtime_installed && status.models.length > 0) {
    return { checkedAt: at, fit: null, reason: 'local models already set up' }
  }

  // The offer names the model the catalog recommends. A machine without a recommendation does not
  // qualify, even when a model would run with part of its weights in system memory.
  const model = catalog.find(candidate => candidate.recommended && candidate.fits)

  return model
    ? { checkedAt: at, fit: { model }, reason: `fits ${model.id}` }
    : { checkedAt: at, fit: null, reason: 'no catalog model is recommended for this machine' }
}

/** One status + catalog read per connection; a transient miss is retried on the next read. */
export function readLocalSetupEligibility(): Promise<Eligibility> {
  const cached = $localSetupEligibility.get()

  return cached && !cached.transient ? Promise.resolve(cached) : refreshLocalSetupEligibility()
}

/** The answer once a connection is up: before that there is no machine to read, so it waits for one. */
export function readConnectedLocalSetupEligibility(): Promise<Eligibility> {
  if ($connection.get()?.mode) {
    return readLocalSetupEligibility()
  }

  return new Promise(resolve => {
    const stop = $connection.listen(connection => {
      if (connection?.mode) {
        stop()
        resolve(readLocalSetupEligibility())
      }
    })
  })
}

/**
 * Whether a configured provider already points at this machine, so a local model is set up outside
 * Hermes' own runtime. Fails open: a provider read that cannot answer is not evidence either way,
 * and must not hide or un-hide the offer on its own.
 */
function readLocalEndpointInUse(): Promise<boolean> {
  return getGlobalModelOptions()
    .then(options => (options.providers ?? []).some(provider => isLoopbackEndpoint(provider.api_url)))
    .catch(() => false)
}

/** Read again, keeping the current answer on screen until the new one lands. */
export function refreshLocalSetupEligibility(): Promise<Eligibility> {
  const mode = $connection.get()?.mode ?? null

  // Settled without a request: the flag, a missing connection, or a remote
  // backend (whose hardware is not this computer's) all answer "no" locally.
  if (!$localModelsEnabled.get() || mode !== 'local') {
    const settled = $localModelsEnabled.get()
      ? pickLocalSetupFit(mode, null, null, false)
      : { checkedAt: Date.now(), fit: null, reason: 'local models are off in this build' }

    $localSetupEligibility.set(settled)

    return Promise.resolve(settled)
  }

  const generation = eligibilityGeneration

  eligibilityRead ??= Promise.all([getLocalModelsStatus(), getLocalCatalog(), readLocalEndpointInUse()])
    .then(([status, catalog, localEndpointInUse]) =>
      pickLocalSetupFit(mode, status, catalog.models, localEndpointInUse)
    )
    .catch((error: Error) => ({
      checkedAt: Date.now(),
      fit: null,
      reason: `eligibility read failed: ${error.message}`,
      transient: true
    }))
    .then(result => {
      // The connection changed while reading: the answer describes a machine this window has left.
      if (generation !== eligibilityGeneration) {
        return { checkedAt: Date.now(), fit: null, reason: 'connection changed while reading', transient: true }
      }

      eligibilityRead = null

      // A failed re-read keeps a good answer on screen rather than hiding the offer on one bad fetch.
      if (!(result.transient && $localSetupEligibility.get()?.fit)) {
        $localSetupEligibility.set(result)
      }

      return result
    })

  return eligibilityRead
}

/** Drop the cached answer: the backend it described is gone (connection or profile changed; debug reset). */
function invalidateLocalSetupEligibility(): void {
  eligibilityGeneration += 1
  eligibilityRead = null
  $localSetupEligibility.set(null)
}

/** Which backend's machine the cached answer describes. Window-state republishes keep it. */
function backendIdentity(): string {
  const connection = $connection.get()

  return connection ? `${connection.mode ?? ''}|${connection.connectionId ?? ''}|${connection.baseUrl}` : ''
}

let lastBackend = backendIdentity()

$connection.listen(() => {
  const next = backendIdentity()

  if (next !== lastBackend) {
    lastBackend = next
    invalidateLocalSetupEligibility()
  }
})

/** The model-menu row stays until setup completes. */
export const $localSetupRowFit = computed($localSetupEligibility, eligibility => eligibility?.fit ?? null)

// ── Debug handle ────────────────────────────────────────────────────────────
// Ships in packaged builds on purpose: rtxspark/vespyr run MSIX bundles we
// drive over CDP, and `state()` is how a run that showed nothing explains itself.

interface LocalSetupDebug {
  state: () => { eligibility: Eligibility | null; localModelsEnabled: boolean }
}

declare global {
  interface Window {
    __hermesTips?: LocalSetupDebug
  }
}

window.__hermesTips = {
  state: () => ({ eligibility: $localSetupEligibility.get(), localModelsEnabled: $localModelsEnabled.get() })
}
