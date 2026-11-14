import { stampRowsWithOwningConnection } from '@/lib/session-owner-stamp'
import type {
  AutomationBlueprint,
  CronCancelResult,
  CronDeliveryTarget,
  CronJob,
  CronJobCreatePayload,
  CronJobUpdates
} from '@/types/cron'
import type { SessionInfo } from '@/types/hermes'

import {
  ambientOwnerConnectionId,
  capabilityScoped,
  connectionScoped,
  hermesApi,
  profileScoped,
  STARTUP_REQUEST_TIMEOUT_MS
} from './client'

/** A routine's owner: its agent (profile) and, when pinned, the machine (connection) serving it. The
 *  per-job calls below take one so a surface listing several agents' routines acts on each in its own
 *  profile; without it they keep the ambient scope, exactly as before. */
export interface CronOwnerScope {
  connectionId?: null | string
  profile?: null | string
}

function cronScoped(owner?: CronOwnerScope) {
  return owner ? capabilityScoped(owner) : { ...profileScoped(), ...connectionScoped() }
}

/** `?profile=` names the owning profile for the backend (a hint it verifies for per-job calls). */
function ownerQuery(owner: CronOwnerScope | undefined, joiner: '?' | '&' = '?'): string {
  const profile = owner?.profile?.trim()

  return profile ? `${joiner}profile=${encodeURIComponent(profile)}` : ''
}

// The cron trigger endpoint intentionally waits for the whole job so its
// response reflects the persisted execution result. Agent jobs can run far
// longer than the Electron fetch default; keep this override local to the one
// synchronous long-operation endpoint rather than weakening all API timeouts.
const CRON_TRIGGER_REQUEST_TIMEOUT_MS = 24 * 60 * 60 * 1000

// Cron jobs are stored per-profile (<HERMES_HOME>/cron/jobs.json), and the
// backend's list endpoint defaults to 'all'. Pass a concrete profile key to
// list just that profile's jobs, or 'all' for the unified cross-profile view.
// Omitting the arg keeps the legacy 'all' default for non-profile callers.
// profileScoped() still rides along for backend-process routing.
export function getCronJobs(profile?: string): Promise<CronJob[]> {
  const suffix = profile ? `?profile=${encodeURIComponent(profile)}` : ''

  return hermesApi<CronJob[]>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs${suffix}`,
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getCronJob(jobId: string): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}`
  })
}

export async function getCronJobRuns(jobId: string, limit = 20, scope?: CronOwnerScope): Promise<SessionInfo[]> {
  const owner = scope?.connectionId || ambientOwnerConnectionId()

  const { runs } = await hermesApi<{ runs: SessionInfo[] }>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/runs?limit=${limit}${ownerQuery(scope, '&')}`
  })

  // Run rows are backend-returned sessions like any other page, so they carry
  // the same ownership contract: tag each with the registry connection that
  // served it so a later resume can name the backend that actually holds the
  // run. Without the tag the row's bare (or backend-only) profile is not an
  // exact owner, and clicking a run — especially over SSH/remote, where the
  // run lives on a host the local pool has never seen — fell to the ambient
  // id-only resume and the transcript never loaded (#82527). The backend
  // stamps `profile` (the job's owning profile); this adds the connection
  // half, exactly as /api/sessions and the sidebar slices do.
  return stampRowsWithOwningConnection(runs ?? [], owner, true)
}

// The single source of truth for cron delivery targets (local + configured
// gateways). Both the manual cron editor and the blueprint dialog use this so
// they never offer a platform that isn't connected. Mirrors the dashboard.
export async function getCronDeliveryTargets(): Promise<CronDeliveryTarget[]> {
  const { targets } = await hermesApi<{ targets: CronDeliveryTarget[] }>({
    ...profileScoped(),
    ...connectionScoped(),
    path: '/api/cron/delivery-targets'
  })

  return targets ?? []
}

/** The same targets for one routine owner: its own profile's gateways, on its machine. */
export async function getOwnerCronDeliveryTargets(scope: CronOwnerScope): Promise<CronDeliveryTarget[]> {
  const { targets } = await hermesApi<{ targets: CronDeliveryTarget[] }>({
    ...cronScoped(scope),
    path: `/api/cron/delivery-targets${ownerQuery(scope)}`
  })

  return targets ?? []
}

export function createCronJob(body: CronJobCreatePayload, scope?: CronOwnerScope): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...cronScoped(scope),
    path: `/api/cron/jobs${ownerQuery(scope)}`,
    method: 'POST',
    body
  })
}

export function updateCronJob(jobId: string, updates: CronJobUpdates, scope?: CronOwnerScope): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}${ownerQuery(scope)}`,
    method: 'PUT',
    body: { updates }
  })
}

export function pauseCronJob(jobId: string, scope?: CronOwnerScope): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/pause${ownerQuery(scope)}`,
    method: 'POST'
  })
}

export function resumeCronJob(jobId: string, scope?: CronOwnerScope): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/resume${ownerQuery(scope)}`,
    method: 'POST'
  })
}

export function triggerCronJob(jobId: string, scope?: CronOwnerScope): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/trigger${ownerQuery(scope)}`,
    method: 'POST',
    timeoutMs: CRON_TRIGGER_REQUEST_TIMEOUT_MS
  })
}

// Asks the job's live run to stop; the run books a 'cancelled' attempt at its next cancel
// check (a second or two), so callers poll the job list for the ledger to settle.
export function cancelCronJobRun(jobId: string, scope?: CronOwnerScope): Promise<CronCancelResult> {
  return hermesApi<CronCancelResult>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/cancel${ownerQuery(scope)}`,
    method: 'POST'
  })
}

export function deleteCronJob(jobId: string, scope?: CronOwnerScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...cronScoped(scope),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}${ownerQuery(scope)}`,
    method: 'DELETE'
  })
}

// Automation Blueprints — parameterized cron templates the backend serves from
// cron/blueprint_catalog.py. getAutomationBlueprints returns the gallery
// (deliver options already rewritten to this machine's configured gateways);
// instantiateAutomationBlueprint fills the slots and creates a real cron job via
// the same create_job path as createCronJob.
//
// Both calls name the target profile via ?profile=: the catalog is per profile
// (built-ins plus that profile's plugin-registered blueprints, deliver options
// from its gateways), and instantiate creates the job in that profile. Fetch the
// catalog for the same profile you instantiate into. Mirrors the dashboard's api.ts.
export function getAutomationBlueprints(
  profile: string,
  connectionId?: null | string
): Promise<{ blueprints: AutomationBlueprint[] }> {
  return hermesApi<{ blueprints: AutomationBlueprint[] }>({
    ...(connectionId ? capabilityScoped({ connectionId, profile }) : { ...profileScoped(), ...connectionScoped() }),
    path: `/api/cron/blueprints?profile=${encodeURIComponent(profile)}`,
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function instantiateAutomationBlueprint(
  body: { blueprint: string; values: Record<string, string> },
  profile: string,
  connectionId?: null | string
): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...(connectionId ? capabilityScoped({ connectionId, profile }) : { ...profileScoped(), ...connectionScoped() }),
    path: `/api/cron/blueprints/instantiate?profile=${encodeURIComponent(profile)}`,
    method: 'POST',
    body
  })
}
