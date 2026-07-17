import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, CircleDollarSign, Gauge, Layers, ListChecks } from "lucide-react";
import { DashboardSection } from "./shell";
import { HealthBadge, KpiCard, MetricGrid, ProgressMetric, StatusPill, type DashboardTone } from "./metrics";
import { DashboardEmptyState } from "./states";
import { cn } from "./utils";

export interface ExecutiveProjectScore {
  id: string;
  name: string;
  owner?: string;
  domain: string;
  status: string;
  tone?: DashboardTone;
  healthScore?: number;
  summary: string;
  metrics?: Array<{ label: string; value: ReactNode }>;
}

export interface ExecutiveActionItem {
  id: string;
  title: string;
  owner: string;
  urgency: "low" | "normal" | "high" | "critical";
  due?: string;
  source?: string;
}

export interface ExecutiveRollupMetric {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: DashboardTone;
}

export interface ExecutiveDomainTab {
  id: string;
  label: string;
  status?: string;
  tone?: DashboardTone;
}

function toneForUrgency(urgency: ExecutiveActionItem["urgency"]): DashboardTone {
  if (urgency === "critical") return "critical";
  if (urgency === "high") return "warning";
  if (urgency === "low") return "neutral";
  return "info";
}

export function ExecutiveHealthRollup({
  metrics,
  className,
}: {
  metrics: ExecutiveRollupMetric[];
  className?: string;
}) {
  return (
    <MetricGrid columns={4} className={className}>
      {metrics.map((metric) => (
        <KpiCard
          key={metric.label}
          label={metric.label}
          value={metric.value}
          detail={metric.detail}
          tone={metric.tone ?? "neutral"}
          icon={metric.tone === "critical" ? AlertTriangle : metric.tone === "success" ? CheckCircle2 : Gauge}
        />
      ))}
    </MetricGrid>
  );
}

export function ExecutiveProjectScorecard({
  project,
  className,
}: {
  project: ExecutiveProjectScore;
  className?: string;
}) {
  return (
    <article className={cn("rounded-lg border border-border bg-card p-4 shadow-sm", className)}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{project.domain}</div>
          <h3 className="mt-1 truncate text-base font-semibold text-foreground">{project.name}</h3>
          {project.owner ? <p className="mt-1 text-xs text-muted-foreground">{project.owner}</p> : null}
        </div>
        <HealthBadge label={project.status} tone={project.tone ?? "unknown"} />
      </div>
      <p className="mt-3 text-sm text-muted-foreground">{project.summary}</p>
      {typeof project.healthScore === "number" ? (
        <div className="mt-4">
          <ProgressMetric label="Health score" value={project.healthScore} tone={project.tone ?? "info"} />
        </div>
      ) : null}
      {project.metrics?.length ? (
        <dl className="mt-4 grid grid-cols-2 gap-3 border-t border-border pt-3">
          {project.metrics.map((metric) => (
            <div key={metric.label} className="min-w-0">
              <dt className="truncate text-xs text-muted-foreground">{metric.label}</dt>
              <dd className="mt-1 truncate text-sm font-semibold text-foreground">{metric.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </article>
  );
}

export function ExecutiveActionQueue({
  id,
  items,
  title = "Action Needed",
  className,
}: {
  id?: string;
  items: ExecutiveActionItem[];
  title?: string;
  className?: string;
}) {
  return (
    <DashboardSection id={id} title={title} description="Cross-project decisions, blockers, and follow-ups." className={className}>
      {items.length ? (
        <div className="space-y-3">
          {items.map((item) => (
            <article key={item.id} className="flex flex-wrap items-start justify-between gap-3 rounded-md border border-border bg-background p-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-foreground">{item.title}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {item.owner}{item.source ? ` · ${item.source}` : ""}{item.due ? ` · due ${item.due}` : ""}
                </div>
              </div>
              <StatusPill tone={toneForUrgency(item.urgency)}>{item.urgency}</StatusPill>
            </article>
          ))}
        </div>
      ) : (
        <DashboardEmptyState title="No executive actions" description="No cross-project action queue items are open." />
      )}
    </DashboardSection>
  );
}

export function ExecutiveCostCapacityRollup({
  cost,
  capacity,
  throughput,
  className,
}: {
  cost: ExecutiveRollupMetric;
  capacity: ExecutiveRollupMetric;
  throughput: ExecutiveRollupMetric;
  className?: string;
}) {
  return (
    <MetricGrid columns={3} className={className}>
      <KpiCard label={cost.label} value={cost.value} detail={cost.detail} tone={cost.tone ?? "info"} icon={CircleDollarSign} />
      <KpiCard label={capacity.label} value={capacity.value} detail={capacity.detail} tone={capacity.tone ?? "warning"} icon={Gauge} />
      <KpiCard label={throughput.label} value={throughput.value} detail={throughput.detail} tone={throughput.tone ?? "success"} icon={ListChecks} />
    </MetricGrid>
  );
}

export function ExecutiveDomainTabs({
  tabs,
  activeId,
  onSelect,
  className,
}: {
  tabs: ExecutiveDomainTab[];
  activeId: string;
  onSelect?: (id: string) => void;
  className?: string;
}) {
  return (
    <div className={cn("flex gap-2 overflow-x-auto rounded-lg border border-border bg-card p-2", className)} role="tablist" aria-label="Business domains">
      {tabs.map((tab) => {
        const active = tab.id === activeId;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active}
            className={cn(
              "inline-flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-sm transition",
              active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
            onClick={() => onSelect?.(tab.id)}
          >
            <Layers className="h-4 w-4" aria-hidden="true" />
            <span>{tab.label}</span>
            {tab.status ? <StatusPill tone={tab.tone ?? "unknown"} className={active ? "border-primary-foreground/30 text-primary-foreground" : ""}>{tab.status}</StatusPill> : null}
          </button>
        );
      })}
    </div>
  );
}
