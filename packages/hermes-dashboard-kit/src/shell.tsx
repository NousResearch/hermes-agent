import type { ComponentType, ReactNode } from "react";
import { cn } from "./utils";

export interface DashboardNavItem {
  id: string;
  label: string;
  href?: string;
  active?: boolean;
  icon?: ComponentType<{ className?: string }>;
  badge?: ReactNode;
  description?: string;
  onClick?: () => void;
}

export function DashboardShell({
  sidebar,
  header,
  children,
  className,
}: {
  sidebar?: ReactNode;
  header?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-h-0 w-full", className)}>
      <div className="grid min-h-0 gap-4 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
        {sidebar ? <aside className="min-w-0">{sidebar}</aside> : null}
        <section className="min-w-0 space-y-4">
          {header}
          <DashboardMain>{children}</DashboardMain>
        </section>
      </div>
    </div>
  );
}

export function DashboardMain({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <main className={cn("min-w-0 space-y-4", className)}>{children}</main>;
}

export function DashboardSection({
  id,
  title,
  description,
  action,
  children,
  className,
}: {
  id?: string;
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={cn("rounded-lg border border-border bg-card p-4 shadow-sm", className)}>
      {(title || description || action) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            {title ? <h2 className="text-base font-semibold text-foreground">{title}</h2> : null}
            {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function DashboardHeader({
  title,
  eyebrow,
  description,
  actions,
  meta,
  className,
}: {
  title: string;
  eyebrow?: string;
  description?: string;
  actions?: ReactNode;
  meta?: ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("flex flex-wrap items-start justify-between gap-4 rounded-lg border border-border bg-card p-4", className)}>
      <div className="min-w-0">
        {eyebrow ? <div className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">{eyebrow}</div> : null}
        <h1 className="truncate text-2xl font-semibold text-foreground">{title}</h1>
        {description ? <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p> : null}
        {meta ? <div className="mt-3 flex flex-wrap gap-2">{meta}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function DashboardSidebar({
  title,
  description,
  items,
  footer,
  className,
}: {
  title: string;
  description?: string;
  items: DashboardNavItem[];
  footer?: ReactNode;
  className?: string;
}) {
  return (
    <nav className={cn("rounded-lg border border-border bg-card p-3", className)} aria-label={title}>
      <div className="border-b border-border pb-3">
        <div className="text-sm font-semibold text-foreground">{title}</div>
        {description ? <div className="mt-1 text-xs text-muted-foreground">{description}</div> : null}
      </div>
      <div className="mt-3 space-y-1">
        {items.map((item) => {
          const Icon = item.icon;
          const content = (
            <>
              {Icon ? <Icon className="h-4 w-4 shrink-0" aria-hidden="true" /> : null}
              <span className="min-w-0 flex-1 truncate">{item.label}</span>
              {item.badge}
            </>
          );
          const classes = cn(
            "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition",
            item.active
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-muted hover:text-foreground",
          );

          return item.href ? (
            <a key={item.id} className={classes} href={item.href} title={item.description ?? item.label}>
              {content}
            </a>
          ) : (
            <button key={item.id} className={classes} onClick={item.onClick} title={item.description ?? item.label} type="button">
              {content}
            </button>
          );
        })}
      </div>
      {footer ? <div className="mt-4 border-t border-border pt-3">{footer}</div> : null}
    </nav>
  );
}

export function DashboardPageTitle({ children, className }: { children: ReactNode; className?: string }) {
  return <h1 className={cn("text-2xl font-semibold text-foreground", className)}>{children}</h1>;
}
