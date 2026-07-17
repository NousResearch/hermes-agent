import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";
import { ExternalLink, LayoutDashboard, Radio, ShieldAlert } from "lucide-react";
import { cn } from "./utils";
import { StatusPill } from "./metrics";
function statusTone(status) {
    if (status === "current" || status === "online")
        return "success";
    if (status === "offline" || status === "missing")
        return "critical";
    if (status === "stale")
        return "warning";
    if (status === "checking")
        return "info";
    return "unknown";
}
function dashboardIssues(dashboard) {
    const issues = [];
    if (!dashboard.productionUrl && !dashboard.url)
        issues.push("missing production URL");
    if (!dashboard.localUrl && !dashboard.url)
        issues.push("missing local URL");
    if (!dashboard.healthUrl)
        issues.push("missing health URL");
    return issues;
}
export function useDashboardHealth(dashboards, { enabled = false, intervalMs = 60_000, } = {}) {
    const [health, setHealth] = useState({});
    const healthTargets = useMemo(() => dashboards.filter((dashboard) => dashboard.healthUrl), [dashboards]);
    useEffect(() => {
        if (!enabled)
            return;
        let cancelled = false;
        let timer;
        async function poll() {
            const checking = {};
            for (const dashboard of dashboards) {
                checking[dashboard.id] = dashboard.healthUrl
                    ? { status: "checking" }
                    : { status: "missing", message: "No health URL configured." };
            }
            if (!cancelled)
                setHealth((current) => ({ ...current, ...checking }));
            const results = await Promise.all(healthTargets.map(async (dashboard) => {
                try {
                    const response = await fetch(dashboard.healthUrl, { cache: "no-store" });
                    return {
                        id: dashboard.id,
                        state: {
                            status: response.ok ? "online" : "offline",
                            checkedAt: new Date().toISOString(),
                            message: response.ok ? "Health check passed." : `Health check returned ${response.status}.`,
                        },
                    };
                }
                catch (error) {
                    return {
                        id: dashboard.id,
                        state: {
                            status: "offline",
                            checkedAt: new Date().toISOString(),
                            message: error instanceof Error ? error.message : "Health check failed.",
                        },
                    };
                }
            }));
            if (!cancelled) {
                setHealth((current) => {
                    const merged = { ...current };
                    for (const result of results)
                        merged[result.id] = result.state;
                    return merged;
                });
                timer = setTimeout(poll, intervalMs);
            }
        }
        void poll();
        return () => {
            cancelled = true;
            if (timer)
                clearTimeout(timer);
        };
    }, [dashboards, enabled, healthTargets, intervalMs]);
    return health;
}
export function DashboardLauncher({ dashboards, currentId, title = "Dashboards", empty, className, pollHealth = false, healthPollIntervalMs = 60_000, }) {
    const health = useDashboardHealth(dashboards, { enabled: pollHealth, intervalMs: healthPollIntervalMs });
    if (!dashboards.length) {
        return (_jsxs("div", { className: cn("rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "mb-3 flex items-center gap-2 text-sm font-semibold text-foreground", children: [_jsx(LayoutDashboard, { className: "h-4 w-4" }), title] }), empty ?? (_jsx("div", { className: "rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground", children: "No dashboard manifests are registered." }))] }));
    }
    return (_jsxs("section", { className: cn("rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "mb-3 flex items-center gap-2 text-sm font-semibold text-foreground", children: [_jsx(LayoutDashboard, { className: "h-4 w-4" }), title] }), _jsx("div", { className: "grid gap-3 md:grid-cols-2 xl:grid-cols-3", children: dashboards.map((dashboard) => {
                    const healthState = health[dashboard.id];
                    const status = dashboard.id === currentId ? "current" : healthState?.status ?? dashboard.status ?? "unknown";
                    const href = dashboard.productionUrl ?? dashboard.url ?? dashboard.localUrl;
                    const issues = dashboardIssues(dashboard);
                    return (_jsxs("article", { className: "rounded-lg border border-border bg-background p-3", children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("div", { className: "truncate text-sm font-medium text-foreground", children: dashboard.label }), dashboard.description ? (_jsx("p", { className: "mt-1 line-clamp-2 text-xs text-muted-foreground", children: dashboard.description })) : null] }), _jsx(StatusPill, { tone: statusTone(status), children: status })] }), healthState?.checkedAt ? (_jsxs("div", { className: "mt-2 font-mono-ui text-[11px] text-muted-foreground", children: ["checked ", new Date(healthState.checkedAt).toLocaleTimeString()] })) : null, healthState?.message ? _jsx("p", { className: "mt-2 text-xs text-muted-foreground", children: healthState.message }) : null, _jsxs("div", { className: "mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground", children: [dashboard.category ? _jsx("span", { children: dashboard.category }) : null, dashboard.owner ? _jsx("span", { children: dashboard.owner }) : null, dashboard.healthUrl ? (_jsxs("span", { className: "inline-flex items-center gap-1", children: [_jsx(Radio, { className: "h-3 w-3" }), "health"] })) : (_jsxs("span", { className: "inline-flex items-center gap-1 text-warning", children: [_jsx(ShieldAlert, { className: "h-3 w-3" }), "no health URL"] }))] }), issues.length ? (_jsx("div", { className: "mt-3 space-y-1 rounded-md border border-warning/30 bg-warning/10 p-2 text-xs text-warning", children: issues.map((issue) => (_jsxs("div", { className: "flex items-center gap-1", children: [_jsx(ShieldAlert, { className: "h-3 w-3" }), issue] }, issue))) })) : null, href ? (_jsxs("a", { className: "mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline", href: href, children: ["Open dashboard", _jsx(ExternalLink, { className: "h-3 w-3" })] })) : (_jsx("div", { className: "mt-3 text-xs font-medium text-muted-foreground", children: "No launch URL configured." }))] }, dashboard.id));
                }) })] }));
}
export function ProjectSwitcher({ projects, currentId, onChange, label = "Project", }) {
    return (_jsxs("label", { className: "inline-flex items-center gap-2 text-sm", children: [_jsx("span", { className: "text-muted-foreground", children: label }), _jsx("select", { className: "h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none focus:border-primary", onChange: (event) => onChange?.(event.target.value), value: currentId, children: projects.map((project) => (_jsx("option", { value: project.id, children: project.label }, project.id))) })] }));
}
//# sourceMappingURL=launcher.js.map