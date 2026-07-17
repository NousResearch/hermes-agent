import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AlertTriangle, CheckCircle2, Clock, Loader2, Play, ShieldAlert, Square } from "lucide-react";
import { cn } from "./utils";
import { StatusPill } from "./metrics";
export function CommandBar({ title = "Command Center", description, actions, className, }) {
    return (_jsxs("section", { className: cn("rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "mb-3 flex flex-wrap items-start justify-between gap-3", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-base font-semibold text-foreground", children: title }), description ? _jsx("p", { className: "mt-1 text-sm text-muted-foreground", children: description }) : null] }), _jsxs(StatusPill, { tone: "info", children: [actions.length, " actions"] })] }), _jsx(ActionButtonGroup, { actions: actions })] }));
}
export function ActionButtonGroup({ actions }) {
    return (_jsx("div", { className: "flex flex-wrap gap-2", children: actions.map((action) => {
            const Icon = action.icon;
            return (_jsxs("button", { className: cn("inline-flex h-9 items-center gap-2 rounded-md border px-3 text-sm font-medium transition", action.tone === "critical"
                    ? "border-destructive/30 bg-destructive/10 text-destructive hover:bg-destructive/15"
                    : "border-border bg-background text-foreground hover:bg-muted", action.disabled && "cursor-not-allowed opacity-50"), disabled: action.disabled, onClick: action.onClick, title: action.description ?? action.label, type: "button", children: [Icon ? _jsx(Icon, { className: "h-4 w-4" }) : null, action.label] }, action.id));
        }) }));
}
export function ActivityTimeline({ events, empty = "No activity yet.", }) {
    if (!events.length)
        return _jsx("div", { className: "rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground", children: empty });
    return (_jsx("ol", { className: "space-y-3", children: events.map((event) => (_jsxs("li", { className: "grid grid-cols-[1rem_minmax(0,1fr)] gap-3", children: [_jsx("span", { className: cn("mt-1 h-2.5 w-2.5 rounded-full", event.tone === "critical" ? "bg-destructive" : event.tone === "success" ? "bg-emerald-500" : event.tone === "warning" ? "bg-amber-500" : "bg-primary") }), _jsxs("div", { className: "min-w-0 border-b border-border pb-3", children: [_jsxs("div", { className: "flex flex-wrap items-center justify-between gap-2", children: [_jsx("div", { className: "font-medium text-foreground", children: event.title }), event.timestamp ? _jsx("div", { className: "font-mono-ui text-xs text-muted-foreground", children: event.timestamp }) : null] }), event.description ? _jsx("p", { className: "mt-1 text-sm text-muted-foreground", children: event.description }) : null] })] }, event.id))) }));
}
function queueTone(status) {
    if (status === "completed")
        return "success";
    if (status === "failed" || status === "blocked")
        return "critical";
    if (status === "stale")
        return "warning";
    if (status === "running")
        return "info";
    return "neutral";
}
function queueIcon(status) {
    if (status === "running")
        return Loader2;
    if (status === "completed")
        return CheckCircle2;
    if (status === "failed" || status === "blocked")
        return AlertTriangle;
    if (status === "stale")
        return ShieldAlert;
    if (status === "queued")
        return Clock;
    return Square;
}
export function QueuePanel({ title = "Queue", items, }) {
    return (_jsxs("section", { className: "rounded-lg border border-border bg-card p-4", children: [_jsxs("div", { className: "mb-3 flex items-center justify-between gap-3", children: [_jsx("h2", { className: "text-base font-semibold text-foreground", children: title }), _jsxs(StatusPill, { tone: "info", children: [items.length, " items"] })] }), _jsx("div", { className: "space-y-2", children: items.length ? items.map((item) => {
                    const Icon = queueIcon(item.status);
                    return (_jsxs("div", { className: "flex items-start justify-between gap-3 rounded-md border border-border bg-background p-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsxs("div", { className: "flex items-center gap-2 text-sm font-medium text-foreground", children: [_jsx(Icon, { className: cn("h-4 w-4", item.status === "running" && "animate-spin") }), item.label] }), item.detail ? _jsx("p", { className: "mt-1 text-xs text-muted-foreground", children: item.detail }) : null] }), _jsx(StatusPill, { tone: queueTone(item.status), children: item.status })] }, item.id));
                }) : _jsx("div", { className: "rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground", children: "No queued work." }) })] }));
}
export function RunStatusPanel({ running, queued, failed, completed, }) {
    return (_jsxs("div", { className: "grid gap-2 sm:grid-cols-4", children: [_jsxs(StatusPill, { tone: "info", children: [_jsx(Play, { className: "mr-1 h-3 w-3" }), running, " running"] }), _jsxs(StatusPill, { tone: "neutral", children: [queued, " queued"] }), _jsxs(StatusPill, { tone: "critical", children: [failed, " failed"] }), _jsxs(StatusPill, { tone: "success", children: [completed, " completed"] })] }));
}
export function AuditEventList({ events }) {
    return _jsx(ActivityTimeline, { events: events, empty: "No audit events." });
}
//# sourceMappingURL=operations.js.map