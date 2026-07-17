import type { ReactNode } from "react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { cn } from "./utils";

export function DashboardEmptyState({
  title = "No data",
  description = "There is nothing to show yet.",
  action,
  className,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-h-32 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card/50 p-6 text-center", className)}>
      <Inbox className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
      <div>
        <div className="text-sm font-medium text-foreground">{title}</div>
        <div className="mt-1 text-sm text-muted-foreground">{description}</div>
      </div>
      {action}
    </div>
  );
}

export function DashboardLoadingState({
  label = "Loading",
  className,
}: {
  label?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex min-h-32 items-center justify-center gap-2 rounded-lg border border-border bg-card/60 p-6 text-sm text-muted-foreground", className)}>
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function DashboardErrorState({
  title = "Unable to load",
  message,
  action,
  className,
}: {
  title?: string;
  message?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm", className)} role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-foreground">{title}</div>
        {message ? <div className="mt-1 text-muted-foreground">{message}</div> : null}
        {action ? <div className="mt-3">{action}</div> : null}
      </div>
    </div>
  );
}
