import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ExternalLink, LayoutDashboard, Radio, ShieldAlert } from "lucide-react";
import { cn } from "./utils";
import { StatusPill } from "./metrics";
function statusTone(status) {
    if (status === "current" || status === "online")
        return "success";
    if (status === "offline" || status === "missing")
        return "critical";
    return "unknown";
}
export function DashboardLauncher({ dashboards, currentId, title = "Dashboards", empty, className, }) {
    if (!dashboards.length) {
        return (_jsxs("div", { className: cn("rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "mb-3 flex items-center gap-2 text-sm font-semibold text-foreground", children: [_jsx(LayoutDashboard, { className: "h-4 w-4" }), title] }), empty ?? (_jsx("div", { className: "rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground", children: "No dashboard manifests are registered." }))] }));
    }
    return (_jsxs("section", { className: cn("rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "mb-3 flex items-center gap-2 text-sm font-semibold text-foreground", children: [_jsx(LayoutDashboard, { className: "h-4 w-4" }), title] }), _jsx("div", { className: "grid gap-3 md:grid-cols-2 xl:grid-cols-3", children: dashboards.map((dashboard) => {
                    const status = dashboard.id === currentId ? "current" : dashboard.status ?? "unknown";
                    const href = dashboard.productionUrl ?? dashboard.url ?? dashboard.localUrl;
                    return (_jsxs("article", { className: "rounded-lg border border-border bg-background p-3", children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("div", { className: "truncate text-sm font-medium text-foreground", children: dashboard.label }), dashboard.description ? (_jsx("p", { className: "mt-1 line-clamp-2 text-xs text-muted-foreground", children: dashboard.description })) : null] }), _jsx(StatusPill, { tone: statusTone(status), children: status })] }), _jsxs("div", { className: "mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground", children: [dashboard.category ? _jsx("span", { children: dashboard.category }) : null, dashboard.owner ? _jsx("span", { children: dashboard.owner }) : null, dashboard.healthUrl ? (_jsxs("span", { className: "inline-flex items-center gap-1", children: [_jsx(Radio, { className: "h-3 w-3" }), "health"] })) : (_jsxs("span", { className: "inline-flex items-center gap-1 text-warning", children: [_jsx(ShieldAlert, { className: "h-3 w-3" }), "no health URL"] }))] }), href ? (_jsxs("a", { className: "mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline", href: href, children: ["Open dashboard", _jsx(ExternalLink, { className: "h-3 w-3" })] })) : null] }, dashboard.id));
                }) })] }));
}
export function ProjectSwitcher({ projects, currentId, onChange, label = "Project", }) {
    return (_jsxs("label", { className: "inline-flex items-center gap-2 text-sm", children: [_jsx("span", { className: "text-muted-foreground", children: label }), _jsx("select", { className: "h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none focus:border-primary", onChange: (event) => onChange?.(event.target.value), value: currentId, children: projects.map((project) => (_jsx("option", { value: project.id, children: project.label }, project.id))) })] }));
}
//# sourceMappingURL=launcher.js.map