import * as React from "react";
import { AlertTriangle, Lock, RotateCw } from "lucide-react";
import { EmptyState, GlassCard, GlassPanel, GlassSkeleton, SectionHeader } from "@/components/glass";
import { InfoDot } from "@/components/tooltip";
import { useCountUp } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import type { Loaded } from "@/lib/api";

/** Every panel knows how to be loading, forbidden, broken or empty. A panel that
 *  cannot render its own failure takes the page down with it. */
export function Panel<T>({
  title, detail, icon, state, action, children, empty, className,
}: {
  title: string; detail?: string;
  icon?: React.ComponentType<{ className?: string }>;
  state: Loaded<T>; action?: React.ReactNode; className?: string;
  children: (data: T) => React.ReactNode;
  empty?: (data: T) => { title: string; detail: string; hint?: string } | null;
}) {
  return (
    <GlassPanel className={cn("p-5", className)}>
      <SectionHeader title={title} detail={detail} icon={icon} action={action} />
      <PanelBody state={state} empty={empty}>{children}</PanelBody>
    </GlassPanel>
  );
}

export function PanelBody<T>({
  state, children, empty,
}: {
  state: Loaded<T>;
  children: (data: T) => React.ReactNode;
  empty?: (data: T) => { title: string; detail: string; hint?: string } | null;
}) {
  if (state.state === "loading") {
    return (
      <div className="space-y-2.5" aria-busy="true" aria-live="polite">
        <span className="sr-only">Loading</span>
        <GlassSkeleton className="h-9 w-full" />
        <GlassSkeleton className="h-9 w-[82%]" />
        <GlassSkeleton className="h-9 w-[64%]" />
      </div>
    );
  }
  if (state.state === "forbidden") {
    return (
      <EmptyState
        icon={Lock}
        title="Not visible to your role"
        detail="A viewer sees operational state. Governance, decisions and cost need an admin token, because the same data answers an auditor's question and an attacker's."
      />
    );
  }
  if (state.state === "error") {
    return <ErrorState message={state.message} />;
  }
  const blank = empty?.(state.data);
  return blank ? <EmptyState {...blank} icon={AlertTriangle} /> : <>{children(state.data)}</>;
}

/** Errors are calm and actionable. The technical detail is available but folded away:
 *  an operator needs to know what to do, and only sometimes why. */
export function ErrorState({ message }: { message: string }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="px-1 py-4">
      <div className="flex items-start gap-3">
        <div className="bg-blocked/10 text-blocked mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg">
          <AlertTriangle className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-ink text-sm font-medium">We couldn’t load this</p>
          <p className="text-ink-faint mt-1 text-[13px] leading-relaxed">
            The control plane answered, but not with what this panel needs. Other panels are
            unaffected.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="glass-solid text-ink hover:border-glass-border-lit inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition-colors"
            >
              <RotateCw className="size-3" /> Retry
            </button>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              className="text-ink-faint hover:text-ink-muted text-[12px] transition-colors"
            >
              {open ? "Hide" : "Show"} diagnostics
            </button>
          </div>
          {open ? (
            <pre className="glass-solid text-ink-muted mt-3 overflow-x-auto rounded-lg p-3 font-mono text-[11px] whitespace-pre-wrap">
              {message}
            </pre>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/** A headline figure. No trend arrow, no percentage: the backend keeps no history, and
 *  a fabricated "+12%" is the fastest way to make a real dashboard untrustworthy. */
export function MetricCard({
  label, value, caption, tone = "neutral", hint, onClick, icon: Icon, source = "ok",
}: {
  label: string; value: number; caption?: string;
  tone?: "neutral" | "running" | "waiting" | "blocked";
  hint?: React.ReactNode; onClick?: () => void;
  icon?: React.ComponentType<{ className?: string }>;
  /** State of the panel this figure is counted from. Anything but "ok" shows "—",
   *  never 0: a deployment whose control plane is unreachable has an unknown number of
   *  agents, and printing 0 states something false rather than admitting the gap.
   *  Loading and failed are kept apart — a figure still arriving has not failed. */
  source?: "ok" | "loading" | "error" | "forbidden";
}) {
  const known = source === "ok";
  const shown = useCountUp(known ? value : 0);
  const toneClass = known ? {
    neutral: "text-ink", running: "text-running", waiting: "text-waiting", blocked: "text-blocked",
  }[tone] : "text-ink-faint";

  const body = (
    <>
      <div className="flex items-start justify-between gap-2">
        <span className="text-ink-faint text-[11px] font-medium tracking-wide uppercase">
          {label}
        </span>
        {hint ? <InfoDot text={hint} /> : Icon ? <Icon className="text-ink-faint/60 size-3.5" /> : null}
      </div>
      <div className={cn("mt-2 text-[28px] leading-none font-semibold", toneClass)}>
        {known ? shown.toLocaleString() : "—"}
      </div>
      <p className="text-ink-faint mt-1.5 text-[12px]">
        {known ? caption
          : source === "loading" ? "reading…"
          : source === "forbidden" ? "not visible to your role"
          : "could not be read"}
      </p>
    </>
  );

  return onClick ? (
    <GlassCard className="w-full p-4 text-left" onClick={onClick} role="button" tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}>
      {body}
    </GlassCard>
  ) : (
    <GlassCard className="p-4" interactive={false}>{body}</GlassCard>
  );
}
