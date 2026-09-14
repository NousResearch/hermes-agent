/* NOVA glass primitives.
 *
 * Deliberately small and CSS-first. Every visual effect here is a class in
 * `index.css`; React contributes structure and, in exactly one place, two custom
 * properties on mousemove. Nothing animates on a JS timer, nothing reads layout
 * during a frame, and nothing renders to a canvas — a control plane that is slow
 * on an ordinary laptop is a control plane people stop opening.
 */
import * as React from "react";
import { cn } from "@/lib/utils";

/* ── Surfaces ─────────────────────────────────────────────────────────────── */

export function GlassPanel({
  className, elevated, solid, ...props
}: React.ComponentProps<"div"> & { elevated?: boolean; solid?: boolean }) {
  return (
    <div
      className={cn(
        "rounded-xl",
        solid ? "glass-solid" : elevated ? "glass-elevated" : "glass",
        className,
      )}
      {...props}
    />
  );
}

/** A panel whose highlight follows the pointer. The effect is intentionally almost
 *  subliminal: it makes the surface feel physically present without becoming a toy. */
export function GlassCard({
  className, interactive = true, children, ...props
}: React.ComponentProps<"div"> & { interactive?: boolean }) {
  const onMove = React.useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    const box = target.getBoundingClientRect();
    target.style.setProperty("--px", `${((event.clientX - box.left) / box.width) * 100}%`);
    target.style.setProperty("--py", `${((event.clientY - box.top) / box.height) * 100}%`);
  }, []);

  return (
    <div
      onMouseMove={onMove}
      className={cn("glass lucent rounded-xl", interactive && "interactive", className)}
      {...props}
    >
      {children}
    </div>
  );
}

/* ── Status ───────────────────────────────────────────────────────────────── */

export type State = "running" | "waiting" | "blocked" | "info" | "neutral";

const STATE_CLASS: Record<State, string> = {
  running: "bg-running",
  waiting: "bg-waiting",
  blocked: "bg-blocked",
  info: "bg-info",
  neutral: "bg-neutral",
};

/** A status dot. Only genuinely live states animate — a pulsing dot on something
 *  static is a lie the interface tells every second. */
export function StatusDot({ state, className }: { state: State; className?: string }) {
  return (
    <span className={cn("relative inline-flex size-2 shrink-0", className)}>
      {state === "running" ? (
        <span className={cn("absolute inline-flex size-full rounded-full opacity-60 pulse-running", STATE_CLASS[state])} />
      ) : null}
      <span
        className={cn(
          "relative inline-flex size-2 rounded-full",
          STATE_CLASS[state],
          state === "waiting" && "pulse-waiting",
        )}
      />
    </span>
  );
}

const TONE: Record<State, string> = {
  running: "text-running border-running/25 bg-running/10",
  waiting: "text-waiting border-waiting/25 bg-waiting/10",
  blocked: "text-blocked border-blocked/25 bg-blocked/10",
  info: "text-info border-info/25 bg-info/10",
  neutral: "text-ink-muted border-glass-border bg-glass-1",
};

export function StatusPill({
  state, children, dot = true, className,
}: { state: State; children: React.ReactNode; dot?: boolean; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap",
        TONE[state],
        className,
      )}
    >
      {dot ? <StatusDot state={state} /> : null}
      {children}
    </span>
  );
}

export function Chip({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "border-glass-border bg-glass-1 text-ink-muted inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium",
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ── Structure ────────────────────────────────────────────────────────────── */

export function SectionHeader({
  title, detail, action, icon: Icon,
}: {
  title: string; detail?: string; action?: React.ReactNode;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="mb-4 flex items-start gap-3">
      {Icon ? (
        <div className="glass-solid mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg">
          <Icon className="text-ink-muted size-4" />
        </div>
      ) : null}
      <div className="min-w-0 flex-1">
        <h2 className="text-ink text-[15px] leading-tight font-semibold tracking-tight">{title}</h2>
        {detail ? <p className="text-ink-faint mt-0.5 text-[13px] leading-snug">{detail}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

/** An empty state always says what the section is for, why it is empty, and what to
 *  do next. "No data" tells an operator nothing and makes them doubt the system. */
export function EmptyState({
  title, detail, hint, icon: Icon,
}: {
  title: string; detail: string; hint?: string;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex flex-col items-center px-6 py-10 text-center">
      <div className="glass-solid mb-4 grid size-11 place-items-center rounded-xl">
        {Icon ? <Icon className="text-ink-faint size-5" /> : null}
      </div>
      <p className="text-ink text-sm font-medium">{title}</p>
      <p className="text-ink-faint mt-1.5 max-w-md text-[13px] leading-relaxed">{detail}</p>
      {hint ? (
        <code className="glass-solid text-ink-muted mt-3 rounded-md px-2 py-1 font-mono text-[11px]">
          {hint}
        </code>
      ) : null}
    </div>
  );
}

export function GlassSkeleton({ className }: { className?: string }) {
  return <div className={cn("glass-solid shimmer rounded-md", className)} />;
}
