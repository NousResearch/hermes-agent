import { Fragment, type ReactNode } from "react";
import { cn } from "./utils";
import { DashboardEmptyState } from "./states";
import { StatusPill, type DashboardTone } from "./metrics";

export function ChartPanel({
  id,
  title,
  description,
  action,
  children,
  className,
}: {
  id?: string;
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={cn("rounded-lg border border-border bg-card p-4 shadow-sm", className)}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{title}</h2>
          {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function SimpleBarChart({
  data,
  valueLabel = "value",
}: {
  data: { label: string; value: number }[];
  valueLabel?: string;
}) {
  if (!data.length) return <DashboardEmptyState title="No chart data" description="No series values are available." />;
  const max = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="space-y-3">
      {data.map((item) => (
        <div key={item.label} className="grid grid-cols-[minmax(5rem,10rem)_minmax(0,1fr)_4rem] items-center gap-3 text-sm">
          <div className="truncate text-muted-foreground">{item.label}</div>
          <div className="h-3 overflow-hidden rounded-full bg-muted">
            <div
              aria-label={`${item.label} ${valueLabel}: ${item.value}`}
              className="h-full rounded-full bg-primary"
              style={{ width: `${Math.max(2, (item.value / max) * 100)}%` }}
            />
          </div>
          <div className="text-right font-mono-ui text-xs text-foreground">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

export function SimpleLineChart({
  data,
  height = 140,
}: {
  data: { label: string; value: number }[];
  height?: number;
}) {
  if (data.length < 2) return <DashboardEmptyState title="Not enough data" description="At least two points are required." />;
  const width = 640;
  const min = Math.min(...data.map((item) => item.value));
  const max = Math.max(...data.map((item) => item.value));
  const span = max - min || 1;
  const points = data.map((item, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((item.value - min) / span) * (height - 12) - 6;
    return `${x},${y}`;
  }).join(" ");
  return (
    <div className="overflow-hidden rounded-md border border-border bg-background p-3">
      <svg aria-label="line chart" className="h-auto w-full" viewBox={`0 0 ${width} ${height}`} role="img">
        <polyline fill="none" points={points} stroke="hsl(var(--primary))" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" />
      </svg>
    </div>
  );
}

export function HeatmapGrid({
  rows,
  columns,
  values,
}: {
  rows: string[];
  columns: string[];
  values: Record<string, number>;
}) {
  const max = Math.max(...Object.values(values), 1);
  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[36rem] gap-1" style={{ gridTemplateColumns: `8rem repeat(${columns.length}, minmax(5rem, 1fr))` }}>
        <div />
        {columns.map((column) => <div key={column} className="px-2 py-1 text-xs font-medium text-muted-foreground">{column}</div>)}
        {rows.map((row) => (
          <Fragment key={row}>
            <div key={`${row}-label`} className="px-2 py-2 text-sm text-muted-foreground">{row}</div>
            {columns.map((column) => {
              const key = `${row}:${column}`;
              const value = values[key] ?? 0;
              const opacity = value > 0 ? 0.15 + (value / max) * 0.55 : 0.06;
              return (
                <div
                  key={key}
                  className="rounded-md border border-border px-2 py-2 text-center font-mono-ui text-sm text-foreground"
                  style={{ backgroundColor: `rgb(20 184 166 / ${opacity})` }}
                >
                  {value}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export function InsightPanel({
  title,
  children,
  tone = "info",
}: {
  title: string;
  children: ReactNode;
  tone?: DashboardTone;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <StatusPill tone={tone}>{tone}</StatusPill>
      </div>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

export function FindingCard({
  title,
  description,
  evidence,
  tone = "info",
}: {
  title: string;
  description?: string;
  evidence?: ReactNode;
  tone?: DashboardTone;
}) {
  return (
    <article className="rounded-md border border-border bg-background p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="font-medium text-foreground">{title}</div>
        <StatusPill tone={tone}>{tone}</StatusPill>
      </div>
      {description ? <p className="mt-2 text-sm text-muted-foreground">{description}</p> : null}
      {evidence ? <div className="mt-3 text-xs text-muted-foreground">{evidence}</div> : null}
    </article>
  );
}

export function RecommendationCard({
  title,
  action,
  confidence,
}: {
  title: string;
  action: string;
  confidence?: number;
}) {
  return (
    <article className="rounded-md border border-border bg-background p-3">
      <div className="font-medium text-foreground">{title}</div>
      <p className="mt-2 text-sm text-muted-foreground">{action}</p>
      {typeof confidence === "number" ? (
        <div className="mt-3 text-xs text-muted-foreground">Confidence: {Math.round(confidence * 100)}%</div>
      ) : null}
    </article>
  );
}
