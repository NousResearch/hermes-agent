import type { ComponentType, ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Circle, HelpCircle, XCircle } from "lucide-react";
import { cn } from "./utils";

export type DashboardTone = "neutral" | "success" | "warning" | "critical" | "info" | "unknown";

const toneClasses: Record<DashboardTone, string> = {
  neutral: "border-border bg-muted text-muted-foreground",
  success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  critical: "border-destructive/30 bg-destructive/10 text-destructive",
  info: "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300",
  unknown: "border-border bg-muted text-muted-foreground",
};

const healthIcon: Record<DashboardTone, ComponentType<{ className?: string }>> = {
  neutral: Circle,
  success: CheckCircle2,
  warning: AlertTriangle,
  critical: XCircle,
  info: Circle,
  unknown: HelpCircle,
};

export function StatusPill({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: DashboardTone;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex h-6 items-center rounded-full border px-2 text-xs font-medium", toneClasses[tone], className)}>
      {children}
    </span>
  );
}

export function HealthBadge({
  label,
  tone = "unknown",
  className,
}: {
  label: string;
  tone?: DashboardTone;
  className?: string;
}) {
  const Icon = healthIcon[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium", toneClasses[tone], className)}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  );
}

export function MetricGrid({
  id,
  children,
  columns = 4,
  className,
}: {
  id?: string;
  children: ReactNode;
  columns?: 2 | 3 | 4 | 5 | 6;
  className?: string;
}) {
  const columnsClass = {
    2: "md:grid-cols-2",
    3: "md:grid-cols-2 xl:grid-cols-3",
    4: "md:grid-cols-2 xl:grid-cols-4",
    5: "md:grid-cols-2 xl:grid-cols-5",
    6: "md:grid-cols-2 xl:grid-cols-6",
  }[columns];
  return <div id={id} className={cn("grid gap-3", columnsClass, className)}>{children}</div>;
}

export function KpiCard({
  label,
  value,
  detail,
  tone = "neutral",
  icon: Icon,
  footer,
  loading = false,
  className,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: DashboardTone;
  icon?: ComponentType<{ className?: string }>;
  footer?: ReactNode;
  loading?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("min-h-32 rounded-lg border border-border bg-card p-4 shadow-sm", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 text-sm text-muted-foreground">{label}</div>
        {Icon ? (
          <span className={cn("rounded-md border p-1.5", toneClasses[tone])}>
            <Icon className="h-4 w-4" aria-hidden="true" />
          </span>
        ) : null}
      </div>
      <div className={cn("mt-3 truncate text-3xl font-semibold text-foreground", loading && "h-9 animate-pulse rounded bg-muted text-transparent")}>
        {loading ? "loading" : value}
      </div>
      {detail ? <div className="mt-2 min-h-5 text-sm text-muted-foreground">{detail}</div> : null}
      {footer ? <div className="mt-3 border-t border-border pt-3 text-xs text-muted-foreground">{footer}</div> : null}
    </div>
  );
}

export function ProgressMetric({
  label,
  value,
  max = 100,
  tone = "info",
  detail,
}: {
  label: string;
  value: number;
  max?: number;
  tone?: DashboardTone;
  detail?: ReactNode;
}) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium text-foreground">{Math.round(pct)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", toneClasses[tone].split(" ").find((part) => part.startsWith("bg-")) ?? "bg-primary")} style={{ width: `${pct}%` }} />
      </div>
      {detail ? <div className="text-xs text-muted-foreground">{detail}</div> : null}
    </div>
  );
}

export function CapacityMeter({
  used,
  limit,
  label = "Capacity",
}: {
  used: number;
  limit: number;
  label?: string;
}) {
  const pct = limit > 0 ? used / limit : 0;
  const tone: DashboardTone = pct >= 0.9 ? "critical" : pct >= 0.75 ? "warning" : "success";
  return (
    <ProgressMetric
      label={label}
      value={used}
      max={limit}
      tone={tone}
      detail={`${used.toLocaleString()} / ${limit.toLocaleString()}`}
    />
  );
}

export function TrendDelta({
  value,
  suffix = "%",
}: {
  value: number;
  suffix?: string;
}) {
  const tone = value > 0 ? "success" : value < 0 ? "critical" : "neutral";
  const sign = value > 0 ? "+" : "";
  return <StatusPill tone={tone}>{sign}{value}{suffix}</StatusPill>;
}
