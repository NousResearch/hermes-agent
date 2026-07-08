// Typed wrappers over the cron admin REST API. Shapes mirror web/src/lib/api.ts
// (the source of truth); only the subset CronPage consumes is declared. The
// cron endpoints are self-scoped: each carries an explicit ?profile= param
// ("all" to list every profile's jobs, a concrete profile for mutations).
import { apiGet, apiPost, apiPut, apiDelete } from "./client";

export interface CronJobRepeat {
  times: number | null;
  completed?: number;
}

export interface CronJob {
  id: string;
  profile?: string | null;
  profile_name?: string | null;
  is_default_profile?: boolean;
  name?: string | null;
  prompt?: string | null;
  script?: string | null;
  skills?: string[] | null;
  schedule?: { kind?: string; expr?: string; run_at?: string; display?: string };
  schedule_display?: string | null;
  repeat?: CronJobRepeat | null;
  enabled: boolean;
  state?: string | null;
  deliver?: string | null;
  model?: string | null;
  provider?: string | null;
  base_url?: string | null;
  no_agent?: boolean | null;
  context_from?: string[] | string | null;
  enabled_toolsets?: string[] | null;
  workdir?: string | null;
  last_run_at?: string | null;
  next_run_at?: string | null;
  last_status?: string | null;
  last_error?: string | null;
  last_delivery_error?: string | null;
}

/** Create/update payload. Every field is optional; the MVP form fills name,
 * prompt, schedule (cron expression) and deliver. */
export interface CronJobMutation {
  name?: string;
  prompt?: string;
  schedule?: string;
  deliver?: string;
  skills?: string[];
  provider?: string | null;
  model?: string | null;
  base_url?: string | null;
  script?: string | null;
  no_agent?: boolean;
  context_from?: string[] | null;
  enabled_toolsets?: string[] | null;
  workdir?: string | null;
}

export interface CronDeliveryTarget {
  id: string;
  name: string;
  home_target_set: boolean;
  home_env_var: string | null;
}

/** GET /api/cron/jobs?profile={profile} → CronJob[] */
export const getCronJobs = (profile = "all") =>
  apiGet<CronJob[]>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`);

/** GET /api/cron/delivery-targets → { targets } */
export const getCronDeliveryTargets = () =>
  apiGet<{ targets: CronDeliveryTarget[] }>("/api/cron/delivery-targets");

/** POST /api/cron/jobs?profile={profile} */
export const createCronJob = (job: CronJobMutation, profile = "default") =>
  apiPost<CronJob>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`, job);

/** PUT /api/cron/jobs/{id}?profile={profile}  body { updates } */
export const updateCronJob = (id: string, updates: CronJobMutation, profile = "default") =>
  apiPut<CronJob>(
    `/api/cron/jobs/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`,
    { updates },
  );

/** DELETE /api/cron/jobs/{id}?profile={profile} */
export const deleteCronJob = (id: string, profile = "default") =>
  apiDelete<{ ok: boolean }>(
    `/api/cron/jobs/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`,
  );

/** POST /api/cron/jobs/{id}/pause?profile={profile} */
export const pauseCronJob = (id: string, profile = "default") =>
  apiPost<CronJob>(
    `/api/cron/jobs/${encodeURIComponent(id)}/pause?profile=${encodeURIComponent(profile)}`,
  );

/** POST /api/cron/jobs/{id}/resume?profile={profile} */
export const resumeCronJob = (id: string, profile = "default") =>
  apiPost<CronJob>(
    `/api/cron/jobs/${encodeURIComponent(id)}/resume?profile=${encodeURIComponent(profile)}`,
  );

/** POST /api/cron/jobs/{id}/trigger?profile={profile} */
export const triggerCronJob = (id: string, profile = "default") =>
  apiPost<CronJob>(
    `/api/cron/jobs/${encodeURIComponent(id)}/trigger?profile=${encodeURIComponent(profile)}`,
  );
