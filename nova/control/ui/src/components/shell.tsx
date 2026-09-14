/* The application shell: atmosphere, navigation, command bar.
 *
 * Navigation lists only what the backend actually serves. The brief suggested
 * Integrations and Organization; neither has an endpoint, and a nav item that leads
 * nowhere is worse than an absent one — it teaches people the product is a mock-up.
 */
import * as React from "react";
import {
  Activity, Blocks, BookOpen, CalendarClock, Boxes, CircleCheck, Command, Gauge,
  LayoutDashboard, ListChecks, Moon, ScrollText, ShieldCheck, Sun, Target, X,
} from "lucide-react";
import { GlassPanel, StatusDot } from "@/components/glass";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/tooltip";
import { cn } from "@/lib/utils";

export type NavItem = {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  /** Admin-only routes are shown but marked, so a viewer learns the surface exists
   *  and why they cannot see it — rather than wondering what they are missing. */
  admin?: boolean;
  count?: number;
  /** Set when the count is something a human must act on. */
  urgent?: boolean;
};

export const NAV: NavItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "agents", label: "Agents", icon: Boxes },
  { id: "objectives", label: "Objectives", icon: Target },
  { id: "work", label: "Work", icon: ListChecks },
  { id: "approvals", label: "Approvals", icon: CircleCheck },
  { id: "activity", label: "Activity", icon: Activity, admin: true },
  { id: "knowledge", label: "Knowledge", icon: BookOpen },
  { id: "channels", label: "Channels", icon: Blocks },
  { id: "automations", label: "Automations", icon: CalendarClock },
  { id: "policies", label: "Policies", icon: ShieldCheck, admin: true },
  { id: "usage", label: "Usage", icon: Gauge, admin: true },
];

/** Two slow light fields behind everything. Fixed, blurred, and never over content —
 *  it is atmosphere, not decoration competing for attention. */
export function Atmosphere() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-[var(--canvas)]" />
      <div
        className="bloom-a absolute -top-[20%] -left-[10%] size-[70vw] rounded-full blur-[120px]"
        style={{ background: "radial-gradient(circle, var(--bloom-a), transparent 66%)" }}
      />
      <div
        className="bloom-b absolute -right-[15%] -bottom-[25%] size-[65vw] rounded-full blur-[130px]"
        style={{ background: "radial-gradient(circle, var(--bloom-b), transparent 66%)" }}
      />
    </div>
  );
}

export function Sidebar({
  route, go, items, tenant, product,
}: {
  route: string; go: (id: string) => void; items: NavItem[];
  tenant?: string; product?: string;
}) {
  return (
    <nav aria-label="Sections" className="flex h-full flex-col gap-1 p-3">
      <div className="mb-4 flex items-center gap-2.5 px-2 pt-1">
        <div
          className="grid size-8 shrink-0 place-items-center rounded-lg text-[13px] font-bold"
          style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
          aria-hidden
        >
          {(product ?? "N").slice(0, 1).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="text-ink truncate text-[13px] leading-tight font-semibold">
            {product ?? "Control Center"}
          </div>
          <div className="text-ink-faint truncate font-mono text-[10px]">{tenant ?? ""}</div>
        </div>
      </div>

      {items.map((item) => {
        const active = route === item.id || route.startsWith(`${item.id}/`);
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => go(item.id)}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group relative flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors",
              active
                ? "glass-solid text-ink"
                : "text-ink-muted hover:text-ink hover:bg-glass-1",
            )}
          >
            {active ? (
              <span
                aria-hidden
                className="absolute top-1/2 left-0 h-4 w-0.5 -translate-y-1/2 rounded-r"
                style={{ background: "var(--accent)" }}
              />
            ) : null}
            <item.icon className="size-4 shrink-0" />
            <span className="truncate">{item.label}</span>
            {item.count ? (
              <span
                className={cn(
                  "ml-auto rounded-full px-1.5 py-0.5 text-[10px] font-semibold",
                  // ink-muted, not ink-faint: at 10px this is small text, so it needs
                  // 4.5:1, and the faint tier sits just under that on the tinted pill.
                  item.urgent
                    ? "bg-waiting/15 text-waiting"
                    : "bg-glass-1 text-ink-muted",
                )}
              >
                {item.count}
              </span>
            ) : null}
          </button>
        );
      })}
    </nav>
  );
}

export function TopBar({
  title, subtitle, runtime, healthy, theme, onToggleTheme, onOpenCommand,
}: {
  title: string; subtitle?: string; runtime?: string; healthy?: boolean;
  theme: "dark" | "light"; onToggleTheme: () => void; onOpenCommand: () => void;
}) {
  return (
    <header className="border-glass-border sticky top-0 z-30 border-b backdrop-blur-xl">
      <div className="flex items-center gap-4 px-6 py-3.5">
        <div className="min-w-0 flex-1">
          <h1 className="text-ink truncate text-[15px] leading-tight font-semibold tracking-tight">
            {title}
          </h1>
          {subtitle ? (
            <p className="text-ink-faint truncate text-[12.5px] leading-snug">{subtitle}</p>
          ) : null}
        </div>

        <button
          type="button"
          onClick={onOpenCommand}
          className="glass text-ink-faint hover:text-ink-muted hidden items-center gap-2 rounded-lg px-3 py-1.5 text-[12px] transition-colors md:flex"
        >
          <Command className="size-3.5" />
          <span>Jump to…</span>
          <kbd className="glass-solid text-ink-faint rounded px-1.5 py-0.5 font-mono text-[10px]">⌘K</kbd>
        </button>

        {runtime ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="glass hidden items-center gap-2 rounded-full px-2.5 py-1 text-[11.5px] sm:flex">
                <StatusDot state={healthy === false ? "blocked" : "running"} />
                <span className="text-ink-muted">{runtime}</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {healthy === false
                ? "The platform cannot reach the runtime. Agents will not start."
                : "The runtime is reachable and serving this tenant."}
            </TooltipContent>
          </Tooltip>
        ) : null}

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          className="glass text-ink-muted hover:text-ink rounded-lg p-2 transition-colors"
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </button>
      </div>
    </header>
  );
}

/** Jump-to palette over sections and whatever the page has already loaded. It searches
 *  real, already-fetched data only — it never queries an endpoint that does not exist. */
export function CommandBar({
  open, onClose, items, onPick,
}: {
  open: boolean; onClose: () => void;
  items: Array<{ id: string; label: string; group: string; hint?: string }>;
  onPick: (id: string) => void;
}) {
  const [query, setQuery] = React.useState("");
  const [cursor, setCursor] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const matches = React.useMemo(() => {
    const needle = query.trim().toLowerCase();
    const pool = needle
      ? items.filter((i) => `${i.label} ${i.group} ${i.hint ?? ""}`.toLowerCase().includes(needle))
      : items;
    return pool.slice(0, 40);
  }, [items, query]);

  React.useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  React.useEffect(() => setCursor(0), [query]);

  if (!open) return null;

  const choose = (index: number) => {
    const picked = matches[index];
    if (picked) {
      onPick(picked.id);
      onClose();
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Jump to"
      className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]"
    >
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
      />
      <GlassPanel elevated className="relative w-full max-w-lg overflow-hidden rounded-2xl">
        <div className="border-glass-border flex items-center gap-3 border-b px-4 py-3">
          <Command className="text-ink-faint size-4 shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, matches.length - 1)); }
              if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
              if (e.key === "Enter") { e.preventDefault(); choose(cursor); }
              if (e.key === "Escape") { e.preventDefault(); onClose(); }
            }}
            placeholder="Jump to a section, agent, objective or channel…"
            aria-label="Search sections and loaded records"
            className="text-ink placeholder:text-ink-faint flex-1 bg-transparent text-sm outline-none"
          />
          <button type="button" onClick={onClose} aria-label="Close" className="text-ink-faint hover:text-ink">
            <X className="size-4" />
          </button>
        </div>
        <div className="max-h-[52vh] overflow-y-auto p-2">
          {matches.length === 0 ? (
            <p className="text-ink-faint px-3 py-6 text-center text-[13px]">
              Nothing loaded matches “{query}”.
            </p>
          ) : (
            matches.map((item, index) => (
              <button
                key={`${item.group}-${item.id}-${item.label}`}
                type="button"
                onMouseEnter={() => setCursor(index)}
                onClick={() => choose(index)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px] transition-colors",
                  index === cursor ? "glass-solid text-ink" : "text-ink-muted",
                )}
              >
                <span className="truncate">{item.label}</span>
                {item.hint ? (
                  <span className="text-ink-faint ml-auto truncate text-[11px]">{item.hint}</span>
                ) : null}
                <span className="text-ink-faint shrink-0 text-[10px] tracking-wide uppercase">
                  {item.group}
                </span>
              </button>
            ))
          )}
        </div>
      </GlassPanel>
    </div>
  );
}

export { ScrollText };
