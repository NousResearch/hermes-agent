/* The shapes the Control API actually returns, captured from a live deployment during
 * the audit. Loose where the API passes runtime data straight through: a strict type on
 * something NOVA does not own would break the dashboard on a runtime upgrade rather
 * than degrade it. */
export type Identity = {
  product_name: string; company_name: string; tenant_id: string;
  welcome?: string; support?: { email?: string; url?: string };
  /** Declared colours. Applied to the interface's own tokens at boot — see app.tsx. */
  theme?: { accent?: string; surface?: string; on_accent?: string };
  /** Bundle-relative paths. Truthy means the control plane can serve the image at
   *  /platform/v1/branding/<kind>; the path itself is not fetched by the browser. */
  logo?: string; favicon?: string;
};
export type Health = {
  platform: { version: string; tenant_id: string };
  runtime: Record<string, any> & { runtime?: string; reachable?: boolean; detail?: string; agent_count?: number };
  bundle: { digest: string; agents: number };
};
export type Agent = {
  id: string; display_name?: string; role?: string; description?: string;
  enabled?: boolean; materialized?: boolean; in_sync?: boolean;
  declared_digest?: string; applied_digest?: string;
  limits?: Record<string, any>;
  approval_required_for?: string[];
  knowledge_sources?: string[];
  model?: Record<string, any>;
};
export type Task = {
  task_id: string; title: string; state: string; runtime_status: string;
  agent_id?: string; created_at?: number; started_at?: number | null;
  completed_at?: number | null; priority?: number; consecutive_failures?: number;
  last_error?: string; needs_attention?: boolean;
};
export type Objective = {
  id: string; title?: string; owner?: string; owner_display_name?: string;
  description?: string; acceptance?: string; enabled?: boolean;
  routing_allowed?: boolean; refusals?: string[]; warnings?: string[];
  state?: string; done?: number; total?: number; steps?: any[];
  /** Step ids the supervisor reports as holding this objective up. */
  blocking?: string[];
};
export type KnowledgeSource = {
  id: string; title?: string; description?: string; classification?: string;
  readable_by?: string[]; indexed?: boolean; documents?: number; chunks?: number; bytes?: number;
};
export type Channel = {
  id: string; provider: string; provider_label?: string; display_name?: string;
  enabled?: boolean; transport?: string; needs_public_endpoint?: boolean;
  verification?: string; caveat?: string; status: string;
  allowed_agents?: string[]; routes?: any[]; approval_required_for?: string[];
  derived_agents?: any[]; required_env?: string[]; missing_by_agent?: Record<string, string[]>;
};
export type Policy = {
  declared?: boolean; enforced?: boolean;
  actions?: Record<string, { tools?: string[]; description?: string; requires_approval?: boolean }>;
  permissions?: Record<string, { tools?: string[]; description?: string }>;
  baseline_tools?: string[];
  agents?: Array<{
    id: string; display_name?: string; allow?: string[]; deny?: string[];
    approval_actions?: string[]; unlisted_tool?: string; has_allowlist?: boolean; warnings?: string[];
  }>;
};
/** One row of GET /platform/v1/decisions.
 *
 *  Spelled out rather than `Record<string, any>`: the loose type let the whole dashboard
 *  read `d.outcome`, which this endpoint has never returned, so both the Approvals
 *  escalation list and the Activity timeline silently rendered as empty/unknown against
 *  real data. The field is `effect`. */
export type Decision = {
  ts: string;
  agent_id: string;
  tool: string;
  effect: "allow" | "deny" | "require_approval" | string;
  reason: string;
  rule: string;
  action: string;
  correlation_id: string;
};
export type Budget = {
  controls?: any[]; advisory?: any[]; recorded?: any[];
  observed?: Array<{
    agent_id: string; available?: boolean; detail?: string; caveats?: string[];
    api_calls?: number; total_tokens?: number; estimated_cost_usd?: number; models?: any[];
  }>;
  observed_caveat?: string; enforcement_classes?: any[];
};

/** GET /platform/v1/tasks/{id} — the runtime's own record of how the work went.
 *
 *  Artifacts deliberately carry no path: the adapter drops the attachment's absolute
 *  host path, which is useless to a browser and useful to an attacker. */
export type TaskDetail = {
  task: Task;
  runs: Array<{
    run_id: number; status: string; outcome: string;
    started_at?: number | null; ended_at?: number | null;
    summary: string; error: string; agent_id: string;
  }>;
  notes: Array<{ author: string; body: string; created_at?: number | null }>;
  artifacts: Array<{
    artifact_id: number; filename: string; content_type: string;
    size_bytes: number; uploaded_by: string; created_at?: number | null;
  }>;
  depends_on: string[];
  blocks: string[];
};

/** GET /platform/v1/automations — recurring work the runtime holds, per agent. */
export type AutomationRun = {
  run_id: string; status: string; claimed_at: string;
  started_at?: string | null; finished_at?: string | null; error: string;
};
export type Automation = {
  automation_id: string; name: string; agent_id: string; agent_display_name?: string;
  schedule_display: string; schedule_kind: string; schedule_expression: string;
  enabled: boolean; state: string;
  next_run_at?: string | null; last_run_at?: string | null;
  last_status: string; last_error: string; failure_streak: number;
  paused_reason: string; created_at?: string | null;
  runs: AutomationRun[];
};
/** Whether anything is actually running an agent's schedules. `running` is "the loop
 *  iterates"; `healthy` is "and it completes ticks without raising". */
export type SchedulerHealth = {
  running: boolean; healthy: boolean;
  heartbeat_age_seconds?: number | null; success_age_seconds?: number | null;
  last_error: string; detail: string;
};
/** What NOVA recorded when it declared an automation. Absent for automations created
 *  outside NOVA — shown as exactly that, rather than given invented provenance. */
export type AutomationGovernance = {
  declared_by: string; declared_at: string; digest: string; reason: string;
  permissions: string[]; knowledge: string[]; channels: string[];
  automation_id: string; tenant_id: string;
};
export type AutomationsPayload = {
  scheduling: boolean;
  automations: Automation[];
  detail?: string;
  scheduler_health?: Record<string, SchedulerHealth>;
  counts?: { total: number; enabled: number; paused: number };
  governance?: Record<string, AutomationGovernance>;
  agents?: Array<{ id: string; display_name: string }>;
};
