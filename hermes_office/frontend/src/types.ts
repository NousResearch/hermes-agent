// TypeScript mirrors of hermes_office/models.py — kept in lockstep by hand.

export type Activity = "offline" | "resting" | "learning" | "talking" | "working";
export type Zone = "rest" | "learn" | "talk" | "work";

export const ACTIVITY_TO_ZONE: Record<Activity, Zone> = {
  offline: "rest",
  resting: "rest",
  learning: "learn",
  talking: "talk",
  working: "work",
};

export interface AvatarStyle {
  sprite_id: string;
  hue: number;
}

export interface Employee {
  id: string;
  department_id: string;
  name: string;
  role: string;
  avatar: AvatarStyle;
  model: string;
  provider?: string | null;
  base_url?: string | null;
  enabled_toolsets: string[];
  skills: string[];
  system_prompt: string;
  runtime: "simulated" | "hermes";
  activity: Activity;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface Department {
  id: string;
  name: string;
  mission: string;
  color: string;
  runtime_default: "simulated" | "hermes";
  employee_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: string;
  department_id?: string | null;
  employee_id?: string | null;
  text: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  result_summary: string;
  tokens_in: number;
  tokens_out: number;
}

export interface ActivityEvent {
  ts: string;
  employee_id: string;
  department_id: string;
  kind: "state_change" | "tool_call" | "tool_result" | "assistant" | "clarify" | "error";
  text: string;
  meta: Record<string, unknown>;
}

export interface ResolvedRole {
  recommended_toolsets: string[];
  recommended_skills: string[];
  model_hint: string | null;
  confidence: number;
  rationale_md: string;
  matched_keywords: string[];
}

export interface HostProfile {
  cores: number;
  ram_gb: number;
  gpus: { name: string; vram_gb: number }[];
  os: string;
}

export interface ModelProfile {
  name: string;
  provider_kind: "local" | "api";
  params_b: number;
  quant_bits: number;
  ctx_tokens: number;
  kv_bytes_per_token: number;
  avg_tokens_per_response: number;
  tps_local: number;
  api_p50_ms_per_token: number;
  usd_per_mtok_in: number;
  usd_per_mtok_out: number;
  rate_limit_rpm: number;
}

export interface CapacityReport {
  host: HostProfile;
  model: ModelProfile;
  employee_count: number;
  recommended_concurrency: number;
  expected_p95_latency_ms: number;
  est_usd_per_hour: number;
  memory_headroom_gb: number;
  notes: string[];
}

export interface Preset {
  id: string;
  label: string;
  label_zh: string;
  pictogram: string;
  default_name: string;
  avatar_sprite: string;
  default_hue: number;
  toolsets: string[];
  skills: string[];
  system_prompt: string;
  summary: string;
}

export interface SkillInfo {
  id: string;
  title: string;
  source: string;
}

export interface ToolsetInfo {
  id: string;
  description: string;
}
