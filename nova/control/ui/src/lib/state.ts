/* One place that maps the backend's vocabulary onto NOVA's five semantic states.
 *
 * Centralised because a status colour that means "running" on one screen and
 * "healthy" on another teaches people to stop trusting colour — and colour is the
 * fastest channel an operations screen has.
 */
import type { State } from "@/components/glass";

const RUNTIME_STATE: Record<string, State> = {
  running: "running",
  ready: "running",
  done: "neutral",
  completed: "neutral",
  archived: "neutral",
  review: "waiting",
  todo: "waiting",
  pending: "waiting",
  scheduled: "waiting",
  blocked: "blocked",
  failed: "blocked",
};

export function taskState(runtimeStatus: string): State {
  return RUNTIME_STATE[String(runtimeStatus).toLowerCase()] ?? "neutral";
}

/** Operational language, not database language. "Awaiting review" is what a person
 *  needs to know; "review" is what the column happens to be called. */
const TASK_LABEL: Record<string, string> = {
  running: "Running",
  ready: "Ready to run",
  todo: "Waiting on earlier work",
  review: "Awaiting review",
  blocked: "Blocked",
  done: "Complete",
  completed: "Complete",
  archived: "Archived",
  failed: "Failed",
  scheduled: "Scheduled",
  pending: "Queued",
};

export function taskLabel(runtimeStatus: string): string {
  const key = String(runtimeStatus).toLowerCase();
  return TASK_LABEL[key] ?? key.replace(/_/g, " ");
}

export function channelState(status: string): State {
  return status === "connected" ? "running"
    : status === "needs_credentials" ? "waiting"
    : "neutral";
}

export function channelLabel(status: string): string {
  return status === "connected" ? "Connected"
    : status === "needs_credentials" ? "Credential required"
    : status === "disabled" ? "Not in service"
    : String(status).replace(/_/g, " ");
}

export function objectiveState(state: string): State {
  const key = String(state).toLowerCase();
  return key === "running" ? "running"
    : key === "blocked" ? "blocked"
    : key === "done" || key === "complete" ? "neutral"
    : "waiting";
}

/** The objective states the supervisor reports, in the sentence case every other pill
 *  on the dashboard uses. Rendering the raw value left one badge lowercase. */
export function objectiveLabel(state: string): string {
  const key = String(state).toLowerCase();
  return key === "running" ? "Running"
    : key === "blocked" ? "Blocked"
    : key === "done" || key === "complete" ? "Complete"
    : key === "ready" ? "Ready"
    : key === "not_submitted" ? "Not submitted"
    : key ? key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " ")
    : "—";
}

export function decisionState(effect: string): State {
  return effect === "deny" ? "blocked"
    : effect === "escalate" || effect === "require_approval" ? "waiting"
    : "running";
}

export function decisionLabel(effect: string): string {
  return effect === "deny" ? "Refused"
    : effect === "escalate" || effect === "require_approval" ? "Sent for approval"
    : effect === "allow" ? "Allowed"
    : String(effect).replace(/_/g, " ");
}

/** Unix seconds → a short relative reading. Absolute time goes in the tooltip; a
 *  timeline is scanned, and "14m ago" is scanned faster than a timestamp. */
export function since(epochSeconds?: number | null): string {
  if (!epochSeconds) return "";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function absolute(epochSeconds?: number | null): string {
  if (!epochSeconds) return "unknown";
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    dateStyle: "medium", timeStyle: "short",
  });
}

/** Initials for an agent avatar.
 *
 *  Not `name.slice(0, 2)`: tenants prefix every agent with the company name, so
 *  "Acme Support Assistant" and "Acme Ops" both reduce to "AC" and every avatar in the
 *  deployment looks identical. Take the first letter of each of the first two words
 *  instead, which is what distinguishes them. */
export function initials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

/** ISO-8601 (what the decisions endpoint returns) → the same short relative reading. */
export function sinceIso(ts?: string | null): string {
  if (!ts) return "";
  const ms = Date.parse(ts);
  return Number.isNaN(ms) ? "" : since(Math.floor(ms / 1000));
}

/** The day an event falls on, for grouping a timeline. "Today" and "Yesterday" read
 *  faster than a date, and a date is still needed for everything older. */
export function dayLabel(ts?: string | null): string {
  if (!ts) return "Undated";
  const ms = Date.parse(ts);
  if (Number.isNaN(ms)) return "Undated";
  const day = new Date(ms); day.setHours(0, 0, 0, 0);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((today.getTime() - day.getTime()) / 86_400_000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  return new Date(ms).toLocaleDateString(undefined,
    { weekday: "short", day: "numeric", month: "short", year: diff > 300 ? "numeric" : undefined });
}

/** ISO-8601 → a full local timestamp, for the tooltip behind a relative reading. */
export function absoluteIso(ts?: string | null): string {
  if (!ts) return "unknown";
  const ms = Date.parse(ts);
  return Number.isNaN(ms) ? "unknown" : absolute(Math.floor(ms / 1000));
}
