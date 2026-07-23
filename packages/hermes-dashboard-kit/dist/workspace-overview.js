import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AlertTriangle, CheckCircle2, Compass, Layers, Route, ShieldAlert } from "lucide-react";
import { DashboardSection } from "./shell";
import { HealthBadge, KpiCard, MetricGrid, StatusPill } from "./metrics";
import { summarizeDashboardSnapshot } from "./contracts";
import { assessDashboardArchitecture } from "./strategy";
import { groupDashboardModulesByWorkspace } from "./workspaces";
function toneFromSeverity(status) {
    if (status === "healthy")
        return "success";
    if (status === "watch")
        return "warning";
    if (status === "degraded" || status === "blocked")
        return "critical";
    return "unknown";
}
function moduleSourceSummary(module) {
    if (!module.dataSources.length)
        return "No declared sources";
    const failing = module.dataSources.filter((source) => source.status === "blocked" || source.status === "degraded").length;
    return failing ? `${failing} source issue${failing === 1 ? "" : "s"}` : `${module.dataSources.length} source${module.dataSources.length === 1 ? "" : "s"}`;
}
export function DashboardWorkspaceOverview({ snapshot, assessment = assessDashboardArchitecture(snapshot), }) {
    const summary = summarizeDashboardSnapshot(snapshot);
    const workspaceGroups = groupDashboardModulesByWorkspace(snapshot.modules);
    return (_jsxs("div", { className: "space-y-4", children: [_jsxs(MetricGrid, { columns: 4, children: [_jsx(KpiCard, { label: "Architecture Status", value: summary.status, tone: toneFromSeverity(summary.status), icon: ShieldAlert }), _jsx(KpiCard, { label: "Workspace Coverage", value: `${assessment.workspaceCoveragePercent}%`, detail: "Six-workspace IA coverage", tone: assessment.workspaceCoveragePercent >= 80 ? "success" : "warning", icon: Compass }), _jsx(KpiCard, { label: "Modules", value: summary.moduleCount, detail: `${summary.degradedSourceCount} degraded sources`, tone: summary.degradedSourceCount ? "warning" : "success", icon: Layers }), _jsx(KpiCard, { label: "Alerts", value: summary.alertCount, detail: `${summary.criticalAlertCount} blocked`, tone: summary.criticalAlertCount ? "critical" : summary.alertCount ? "warning" : "success", icon: AlertTriangle })] }), _jsx(DashboardSection, { title: "Workspace Map", description: "Use this to collapse dashboard sprawl into the six Hermes operating workspaces.", children: _jsx("div", { className: "grid gap-3 xl:grid-cols-3", children: workspaceGroups.map((workspace) => (_jsxs("article", { className: "rounded-lg border border-border bg-background p-3", children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("h3", { className: "text-sm font-semibold text-foreground", children: workspace.label }), _jsx("p", { className: "mt-1 text-xs text-muted-foreground", children: workspace.primaryQuestion })] }), _jsx(StatusPill, { tone: workspace.modules.length ? "success" : "unknown", children: workspace.modules.length })] }), _jsx("div", { className: "mt-3 space-y-2", children: workspace.modules.length ? workspace.modules.map((module) => (_jsxs("div", { className: "rounded-md border border-border p-2", children: [_jsxs("div", { className: "flex items-center justify-between gap-2", children: [_jsx("span", { className: "truncate text-sm font-medium text-foreground", children: module.label }), _jsx(HealthBadge, { label: module.status, tone: toneFromSeverity(module.status) })] }), _jsx("p", { className: "mt-1 text-xs text-muted-foreground", children: module.primaryQuestion }), _jsxs("div", { className: "mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground", children: [_jsx("span", { children: moduleSourceSummary(module) }), module.route ? _jsxs("span", { className: "inline-flex items-center gap-1", children: [_jsx(Route, { className: "h-3 w-3" }), module.route] }) : null] })] }, module.id))) : (_jsx("div", { className: "rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground", children: "No modules assigned yet." })) })] }, workspace.id))) }) }), _jsx(DashboardSection, { title: "Capability Gaps", description: "The checklist that separates visual inspiration from a dashboard that tells the truth.", children: _jsx("div", { className: "grid gap-3 lg:grid-cols-2", children: assessment.capabilities.map((capability) => (_jsxs("article", { className: "rounded-lg border border-border bg-background p-3", children: [_jsxs("div", { className: "flex items-center justify-between gap-3", children: [_jsx("h3", { className: "text-sm font-semibold text-foreground", children: capability.label }), _jsx(HealthBadge, { label: capability.status, tone: toneFromSeverity(capability.status) })] }), _jsx("p", { className: "mt-2 text-sm text-muted-foreground", children: capability.gap }), _jsxs("p", { className: "mt-2 text-sm text-foreground", children: [_jsx(CheckCircle2, { className: "mr-1 inline h-4 w-4" }), capability.nextAction] })] }, capability.id))) }) })] }));
}
//# sourceMappingURL=workspace-overview.js.map