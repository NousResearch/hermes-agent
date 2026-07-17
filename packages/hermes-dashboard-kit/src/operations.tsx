import type { ComponentType } from "react";
import { AlertTriangle, CheckCircle2, Clock, Loader2, Play, ShieldAlert, Square } from "lucide-react";
import { cn } from "./utils";
import { StatusPill, type DashboardTone } from "./metrics";

export interface CommandAction {
  id: string;
  label: string;
  description?: string;
  icon?: ComponentType<{ className?: string }>;
  tone?: DashboardTone;
  disabled?: boolean;
  disabledReason?: string;
  permission?: "viewer" | "operator" | "admin";
  requiresConfirmation?: boolean;
  riskLevel?: "low" | "medium" | "high";
  onClick?: () => void;
}

export function CommandBar({
  title = "Command Center",
  description,
  actions,
  className,
}: {
  title?: string;
  description?: string;
  actions: CommandAction[];
  className?: string;
}) {
  return (
    <section className={cn("rounded-lg border border-border bg-card p-4", className)}>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{title}</h2>
          {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
        </div>
        <StatusPill tone="info">{actions.length} actions</StatusPill>
      </div>
      <ActionButtonGroup actions={actions} />
    </section>
  );
}

export function ActionButtonGroup({ actions }: { actions: CommandAction[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {actions.map((action) => {
        const Icon = action.icon;
        const isHighRisk = action.riskLevel === "high" || action.requiresConfirmation || action.tone === "critical";
        const title = action.disabledReason ?? action.description ?? action.label;
        return (
          <div key={action.id} className="inline-flex flex-col gap-1">
            <button
              aria-describedby={action.disabledReason ? `${action.id}-reason` : undefined}
              className={cn(
                "inline-flex h-9 items-center gap-2 rounded-md border px-3 text-sm font-medium transition",
                action.tone === "critical"
                  ? "border-destructive/30 bg-destructive/10 text-destructive hover:bg-destructive/15"
                  : action.tone === "warning" || action.riskLevel === "medium"
                    ? "border-warning/30 bg-warning/10 text-warning hover:bg-warning/15"
                    : "border-border bg-background text-foreground hover:bg-muted",
                action.disabled && "cursor-not-allowed opacity-50",
              )}
              disabled={action.disabled}
              onClick={action.onClick}
              title={title}
              type="button"
            >
              {Icon ? <Icon className="h-4 w-4" /> : null}
              {action.label}
              {action.permission ? <span className="rounded-sm bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">{action.permission}</span> : null}
              {isHighRisk ? <ShieldAlert className="h-3.5 w-3.5" aria-label="high risk action" /> : null}
            </button>
            {action.disabledReason ? (
              <span id={`${action.id}-reason`} className="max-w-44 text-xs text-muted-foreground">
                {action.disabledReason}
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

export interface ActivityEvent {
  id: string;
  title: string;
  timestamp?: string;
  description?: string;
  tone?: DashboardTone;
}

export function ActivityTimeline({
  events,
  empty = "No activity yet.",
}: {
  events: ActivityEvent[];
  empty?: string;
}) {
  if (!events.length) return <div className="rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">{empty}</div>;
  return (
    <ol className="space-y-3">
      {events.map((event) => (
        <li key={event.id} className="grid grid-cols-[1rem_minmax(0,1fr)] gap-3">
          <span className={cn("mt-1 h-2.5 w-2.5 rounded-full", event.tone === "critical" ? "bg-destructive" : event.tone === "success" ? "bg-emerald-500" : event.tone === "warning" ? "bg-amber-500" : "bg-primary")} />
          <div className="min-w-0 border-b border-border pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-medium text-foreground">{event.title}</div>
              {event.timestamp ? <div className="font-mono-ui text-xs text-muted-foreground">{event.timestamp}</div> : null}
            </div>
            {event.description ? <p className="mt-1 text-sm text-muted-foreground">{event.description}</p> : null}
          </div>
        </li>
      ))}
    </ol>
  );
}

export type QueueStatus = "queued" | "running" | "completed" | "failed" | "blocked" | "stale";

export interface QueueItem {
  id: string;
  label: string;
  status: QueueStatus;
  detail?: string;
}

function queueTone(status: QueueStatus): DashboardTone {
  if (status === "completed") return "success";
  if (status === "failed" || status === "blocked") return "critical";
  if (status === "stale") return "warning";
  if (status === "running") return "info";
  return "neutral";
}

function queueIcon(status: QueueStatus) {
  if (status === "running") return Loader2;
  if (status === "completed") return CheckCircle2;
  if (status === "failed" || status === "blocked") return AlertTriangle;
  if (status === "stale") return ShieldAlert;
  if (status === "queued") return Clock;
  return Square;
}

export function QueuePanel({
  title = "Queue",
  items,
}: {
  title?: string;
  items: QueueItem[];
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <StatusPill tone="info">{items.length} items</StatusPill>
      </div>
      <div className="space-y-2">
        {items.length ? items.map((item) => {
          const Icon = queueIcon(item.status);
          return (
            <div key={item.id} className="flex items-start justify-between gap-3 rounded-md border border-border bg-background p-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Icon className={cn("h-4 w-4", item.status === "running" && "animate-spin")} />
                  {item.label}
                </div>
                {item.detail ? <p className="mt-1 text-xs text-muted-foreground">{item.detail}</p> : null}
              </div>
              <StatusPill tone={queueTone(item.status)}>{item.status}</StatusPill>
            </div>
          );
        }) : <div className="rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">No queued work.</div>}
      </div>
    </section>
  );
}

export function RunStatusPanel({
  running,
  queued,
  failed,
  completed,
}: {
  running: number;
  queued: number;
  failed: number;
  completed: number;
}) {
  return (
    <div className="grid gap-2 sm:grid-cols-4">
      <StatusPill tone="info"><Play className="mr-1 h-3 w-3" />{running} running</StatusPill>
      <StatusPill tone="neutral">{queued} queued</StatusPill>
      <StatusPill tone="critical">{failed} failed</StatusPill>
      <StatusPill tone="success">{completed} completed</StatusPill>
    </div>
  );
}

export function AuditEventList({ events }: { events: ActivityEvent[] }) {
  return <ActivityTimeline events={events} empty="No audit events." />;
}
