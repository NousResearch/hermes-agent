import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AlertTriangle, CheckCircle2, CircleDollarSign, Gauge, Layers, ListChecks } from "lucide-react";
import { DashboardSection } from "./shell";
import { HealthBadge, KpiCard, MetricGrid, ProgressMetric, StatusPill } from "./metrics";
import { DashboardEmptyState } from "./states";
import { cn } from "./utils";
function toneForUrgency(urgency) {
    if (urgency === "critical")
        return "critical";
    if (urgency === "high")
        return "warning";
    if (urgency === "low")
        return "neutral";
    return "info";
}
export function ExecutiveHealthRollup({ metrics, className, }) {
    return (_jsx(MetricGrid, { columns: 4, className: className, children: metrics.map((metric) => (_jsx(KpiCard, { label: metric.label, value: metric.value, detail: metric.detail, tone: metric.tone ?? "neutral", icon: metric.tone === "critical" ? AlertTriangle : metric.tone === "success" ? CheckCircle2 : Gauge }, metric.label))) }));
}
export function ExecutiveProjectScorecard({ project, className, }) {
    return (_jsxs("article", { className: cn("rounded-lg border border-border bg-card p-4 shadow-sm", className), children: [_jsxs("div", { className: "flex flex-wrap items-start justify-between gap-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("div", { className: "text-xs font-medium uppercase tracking-wide text-muted-foreground", children: project.domain }), _jsx("h3", { className: "mt-1 truncate text-base font-semibold text-foreground", children: project.name }), project.owner ? _jsx("p", { className: "mt-1 text-xs text-muted-foreground", children: project.owner }) : null] }), _jsx(HealthBadge, { label: project.status, tone: project.tone ?? "unknown" })] }), _jsx("p", { className: "mt-3 text-sm text-muted-foreground", children: project.summary }), typeof project.healthScore === "number" ? (_jsx("div", { className: "mt-4", children: _jsx(ProgressMetric, { label: "Health score", value: project.healthScore, tone: project.tone ?? "info" }) })) : null, project.metrics?.length ? (_jsx("dl", { className: "mt-4 grid grid-cols-2 gap-3 border-t border-border pt-3", children: project.metrics.map((metric) => (_jsxs("div", { className: "min-w-0", children: [_jsx("dt", { className: "truncate text-xs text-muted-foreground", children: metric.label }), _jsx("dd", { className: "mt-1 truncate text-sm font-semibold text-foreground", children: metric.value })] }, metric.label))) })) : null] }));
}
export function ExecutiveActionQueue({ id, items, title = "Action Needed", className, }) {
    return (_jsx(DashboardSection, { id: id, title: title, description: "Cross-project decisions, blockers, and follow-ups.", className: className, children: items.length ? (_jsx("div", { className: "space-y-3", children: items.map((item) => (_jsxs("article", { className: "flex flex-wrap items-start justify-between gap-3 rounded-md border border-border bg-background p-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("div", { className: "truncate text-sm font-medium text-foreground", children: item.title }), _jsxs("div", { className: "mt-1 text-xs text-muted-foreground", children: [item.owner, item.source ? ` · ${item.source}` : "", item.due ? ` · due ${item.due}` : ""] })] }), _jsx(StatusPill, { tone: toneForUrgency(item.urgency), children: item.urgency })] }, item.id))) })) : (_jsx(DashboardEmptyState, { title: "No executive actions", description: "No cross-project action queue items are open." })) }));
}
export function ExecutiveCostCapacityRollup({ cost, capacity, throughput, className, }) {
    return (_jsxs(MetricGrid, { columns: 3, className: className, children: [_jsx(KpiCard, { label: cost.label, value: cost.value, detail: cost.detail, tone: cost.tone ?? "info", icon: CircleDollarSign }), _jsx(KpiCard, { label: capacity.label, value: capacity.value, detail: capacity.detail, tone: capacity.tone ?? "warning", icon: Gauge }), _jsx(KpiCard, { label: throughput.label, value: throughput.value, detail: throughput.detail, tone: throughput.tone ?? "success", icon: ListChecks })] }));
}
export function ExecutiveDomainTabs({ tabs, activeId, onSelect, className, }) {
    return (_jsx("div", { className: cn("flex gap-2 overflow-x-auto rounded-lg border border-border bg-card p-2", className), role: "tablist", "aria-label": "Business domains", children: tabs.map((tab) => {
            const active = tab.id === activeId;
            return (_jsxs("button", { type: "button", role: "tab", "aria-selected": active, className: cn("inline-flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-sm transition", active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"), onClick: () => onSelect?.(tab.id), children: [_jsx(Layers, { className: "h-4 w-4", "aria-hidden": "true" }), _jsx("span", { children: tab.label }), tab.status ? _jsx(StatusPill, { tone: tab.tone ?? "unknown", className: active ? "border-primary-foreground/30 text-primary-foreground" : "", children: tab.status }) : null] }, tab.id));
        }) }));
}
//# sourceMappingURL=executive.js.map