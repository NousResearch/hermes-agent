import type { ReactNode } from "react";
import { ExternalLink, LayoutDashboard, Radio, ShieldAlert } from "lucide-react";
import { cn } from "./utils";
import { StatusPill, type DashboardTone } from "./metrics";

export interface DashboardRegistryEntry {
  id: string;
  label: string;
  description?: string;
  url?: string;
  localUrl?: string;
  productionUrl?: string;
  healthUrl?: string;
  status?: "current" | "online" | "offline" | "unknown" | "missing";
  category?: string;
  owner?: string;
}

function statusTone(status: DashboardRegistryEntry["status"]): DashboardTone {
  if (status === "current" || status === "online") return "success";
  if (status === "offline" || status === "missing") return "critical";
  return "unknown";
}

export function DashboardLauncher({
  dashboards,
  currentId,
  title = "Dashboards",
  empty,
  className,
}: {
  dashboards: DashboardRegistryEntry[];
  currentId?: string;
  title?: string;
  empty?: ReactNode;
  className?: string;
}) {
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
          const status = dashboard.id === currentId ? "current" : dashboard.status ?? "unknown";
          const href = dashboard.productionUrl ?? dashboard.url ?? dashboard.localUrl;
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
              {href ? (
                <a className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" href={href}>
                  Open dashboard
                  <ExternalLink className="h-3 w-3" />
                </a>
              ) : null}
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
  projects: { id: string; label: string; status?: DashboardRegistryEntry["status"] }[];
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
