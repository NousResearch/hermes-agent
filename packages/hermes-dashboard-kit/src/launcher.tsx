import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ExternalLink, LayoutDashboard, Radio, ShieldAlert } from "lucide-react";
import { cn } from "./utils";
import { StatusPill, type DashboardTone } from "./metrics";

export type DashboardHealthStatus = "current" | "online" | "offline" | "stale" | "checking" | "unknown" | "missing";

export interface DashboardRegistryEntry {
  id: string;
  label: string;
  description?: string;
  url?: string;
  localUrl?: string;
  productionUrl?: string;
  healthUrl?: string;
  snapshotUrl?: string;
  status?: DashboardHealthStatus;
  category?: string;
  owner?: string;
}

export interface DashboardHealthState {
  status: DashboardHealthStatus;
  checkedAt?: string;
  message?: string;
}

function statusTone(status: DashboardHealthStatus | undefined): DashboardTone {
  if (status === "current" || status === "online") return "success";
  if (status === "offline" || status === "missing") return "critical";
  if (status === "stale") return "warning";
  if (status === "checking") return "info";
  return "unknown";
}

function dashboardIssues(dashboard: DashboardRegistryEntry) {
  const issues: string[] = [];
  if (!dashboard.productionUrl && !dashboard.url) issues.push("missing production URL");
  if (!dashboard.localUrl && !dashboard.url) issues.push("missing local URL");
  if (!dashboard.healthUrl) issues.push("missing health URL");
  return issues;
}

export function useDashboardHealth(
  dashboards: DashboardRegistryEntry[],
  {
    enabled = false,
    intervalMs = 60_000,
  }: {
    enabled?: boolean;
    intervalMs?: number;
  } = {},
) {
  const [health, setHealth] = useState<Record<string, DashboardHealthState>>({});
  const healthTargets = useMemo(() => dashboards.filter((dashboard) => dashboard.healthUrl), [dashboards]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      const checking: Record<string, DashboardHealthState> = {};
      for (const dashboard of dashboards) {
        checking[dashboard.id] = dashboard.healthUrl
          ? { status: "checking" }
          : { status: "missing", message: "No health URL configured." };
      }
      if (!cancelled) setHealth((current) => ({ ...current, ...checking }));

      const results = await Promise.all(
        healthTargets.map(async (dashboard) => {
          try {
            const response = await fetch(dashboard.healthUrl as string, { cache: "no-store" });
            return {
              id: dashboard.id,
              state: {
                status: response.ok ? "online" : "offline",
                checkedAt: new Date().toISOString(),
                message: response.ok ? "Health check passed." : `Health check returned ${response.status}.`,
              } satisfies DashboardHealthState,
            };
          } catch (error) {
            return {
              id: dashboard.id,
              state: {
                status: "offline",
                checkedAt: new Date().toISOString(),
                message: error instanceof Error ? error.message : "Health check failed.",
              } satisfies DashboardHealthState,
            };
          }
        }),
      );

      if (!cancelled) {
        setHealth((current) => {
          const merged = { ...current };
          for (const result of results) merged[result.id] = result.state;
          return merged;
        });
        timer = setTimeout(poll, intervalMs);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [dashboards, enabled, healthTargets, intervalMs]);

  return health;
}

export function DashboardLauncher({
  dashboards,
  currentId,
  title = "Dashboards",
  empty,
  className,
  pollHealth = false,
  healthPollIntervalMs = 60_000,
}: {
  dashboards: DashboardRegistryEntry[];
  currentId?: string;
  title?: string;
  empty?: ReactNode;
  className?: string;
  pollHealth?: boolean;
  healthPollIntervalMs?: number;
}) {
  const health = useDashboardHealth(dashboards, { enabled: pollHealth, intervalMs: healthPollIntervalMs });

  if (!dashboards.length) {
    return (
      <div className={cn("rounded-lg border border-border bg-card p-4", className)}>
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
          <LayoutDashboard className="h-4 w-4" />
          {title}
        </div>
        {empty ?? (
          <div className="rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">
            No dashboard manifests are registered.
          </div>
        )}
      </div>
    );
  }

  return (
    <section className={cn("rounded-lg border border-border bg-card p-4", className)}>
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
        <LayoutDashboard className="h-4 w-4" />
        {title}
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {dashboards.map((dashboard) => {
          const healthState = health[dashboard.id];
          const status = dashboard.id === currentId ? "current" : healthState?.status ?? dashboard.status ?? "unknown";
          const href = dashboard.productionUrl ?? dashboard.url ?? dashboard.localUrl;
          const issues = dashboardIssues(dashboard);
          return (
            <article key={dashboard.id} className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-foreground">{dashboard.label}</div>
                  {dashboard.description ? (
                    <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{dashboard.description}</p>
                  ) : null}
                </div>
                <StatusPill tone={statusTone(status)}>{status}</StatusPill>
              </div>
              {healthState?.checkedAt ? (
                <div className="mt-2 font-mono-ui text-[11px] text-muted-foreground">
                  checked {new Date(healthState.checkedAt).toLocaleTimeString()}
                </div>
              ) : null}
              {healthState?.message ? <p className="mt-2 text-xs text-muted-foreground">{healthState.message}</p> : null}
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                {dashboard.category ? <span>{dashboard.category}</span> : null}
                {dashboard.owner ? <span>{dashboard.owner}</span> : null}
                {dashboard.healthUrl ? (
                  <span className="inline-flex items-center gap-1">
                    <Radio className="h-3 w-3" />
                    health
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-warning">
                    <ShieldAlert className="h-3 w-3" />
                    no health URL
                  </span>
                )}
              </div>
              {issues.length ? (
                <div className="mt-3 space-y-1 rounded-md border border-warning/30 bg-warning/10 p-2 text-xs text-warning">
                  {issues.map((issue) => (
                    <div key={issue} className="flex items-center gap-1">
                      <ShieldAlert className="h-3 w-3" />
                      {issue}
                    </div>
                  ))}
                </div>
              ) : null}
              {href ? (
                <a className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" href={href}>
                  Open dashboard
                  <ExternalLink className="h-3 w-3" />
                </a>
              ) : (
                <div className="mt-3 text-xs font-medium text-muted-foreground">No launch URL configured.</div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function ProjectSwitcher({
  projects,
  currentId,
  onChange,
  label = "Project",
}: {
  projects: { id: string; label: string; status?: DashboardHealthStatus }[];
  currentId?: string;
  onChange?: (id: string) => void;
  label?: string;
}) {
  return (
    <label className="inline-flex items-center gap-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <select
        className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none focus:border-primary"
        onChange={(event) => onChange?.(event.target.value)}
        value={currentId}
      >
        {projects.map((project) => (
          <option key={project.id} value={project.id}>
            {project.label}
          </option>
        ))}
      </select>
    </label>
  );
}
