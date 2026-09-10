import type {
  AutomationBlueprint,
  CronDeliveryTarget,
  CronJob,
  CronJobCreatePayload,
  CronJobUpdates,
  SessionInfo
} from '@/types/hermes'

import { connectionScoped, hermesApi, profileScoped, STARTUP_REQUEST_TIMEOUT_MS } from './client'

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

// Cron mutations target ONE per-profile store. The list aggregator stamps each
// job's owning profile (CronJob.profile); pass it here so the backend resolves
// the right store instead of falling back to the caller's ambient profile,
// which from a cross-profile view triggers/pauses/deletes the wrong store or
// 404s with "Job not found". A concrete profile becomes ?profile=<name>; omit
// it to keep the legacy ambient-profile behavior for single-profile callers.
function profileQuery(profile?: string): string {
  return profile ? `?profile=${encodeURIComponent(profile)}` : ''
}

export async function getCronJobRuns(jobId: string, limit = 20, profile?: string): Promise<SessionInfo[]> {
  const query = profile ? `?profile=${encodeURIComponent(profile)}&limit=${limit}` : `?limit=${limit}`

  const { runs } = await hermesApi<{ runs: SessionInfo[] }>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/runs${query}`
  })

  return runs ?? []
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

export function createCronJob(body: CronJobCreatePayload): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: '/api/cron/jobs',
    method: 'POST',
    body
  })
}

export function updateCronJob(jobId: string, updates: CronJobUpdates, profile?: string): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}${profileQuery(profile)}`,
    method: 'PUT',
    body: { updates }
  })
}

export function pauseCronJob(jobId: string, profile?: string): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/pause${profileQuery(profile)}`,
    method: 'POST'
  })
}

export function resumeCronJob(jobId: string, profile?: string): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/resume${profileQuery(profile)}`,
    method: 'POST'
  })
}

export function triggerCronJob(jobId: string, profile?: string): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/trigger${profileQuery(profile)}`,
    method: 'POST',
    timeoutMs: CRON_TRIGGER_REQUEST_TIMEOUT_MS
  })
}

export function deleteCronJob(jobId: string, profile?: string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}${profileQuery(profile)}`,
    method: 'DELETE'
  })
}

// Automation Blueprints — parameterized cron templates the backend serves from
// cron/blueprint_catalog.py. getAutomationBlueprints returns the gallery
// (deliver options already rewritten to this machine's configured gateways);
// instantiateAutomationBlueprint fills the slots and creates a real cron job via
// the same create_job path as createCronJob.
//
// Profile-scoping is intentionally asymmetric: the GET catalog is global (the
// list endpoint takes no profile — only deliver options are rewritten from the
// configured gateways), so it carries only the profileScoped() header for
// routing. instantiate creates a real per-profile job, so it names the target
// profile explicitly via ?profile=. This mirrors the dashboard's api.ts.
export function getAutomationBlueprints(): Promise<{ blueprints: AutomationBlueprint[] }> {
  return hermesApi<{ blueprints: AutomationBlueprint[] }>({
    ...profileScoped(),
    ...connectionScoped(),
    path: '/api/cron/blueprints',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function instantiateAutomationBlueprint(
  body: { blueprint: string; values: Record<string, string> },
  profile: string
): Promise<CronJob> {
  return hermesApi<CronJob>({
    ...profileScoped(),
    ...connectionScoped(),
    path: `/api/cron/blueprints/instantiate?profile=${encodeURIComponent(profile)}`,
    method: 'POST',
    body
  })
}
