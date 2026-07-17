import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AlertTriangle, CheckCircle2, Circle, HelpCircle, XCircle } from "lucide-react";
import { cn } from "./utils";
const toneClasses = {
    neutral: "border-border bg-muted text-muted-foreground",
    success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    critical: "border-destructive/30 bg-destructive/10 text-destructive",
    info: "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    unknown: "border-border bg-muted text-muted-foreground",
};
const healthIcon = {
    neutral: Circle,
    success: CheckCircle2,
    warning: AlertTriangle,
    critical: XCircle,
    info: Circle,
    unknown: HelpCircle,
};
export function StatusPill({ children, tone = "neutral", className, }) {
    return (_jsx("span", { className: cn("inline-flex h-6 items-center rounded-full border px-2 text-xs font-medium", toneClasses[tone], className), children: children }));
}
export function HealthBadge({ label, tone = "unknown", className, }) {
    const Icon = healthIcon[tone];
    return (_jsxs("span", { className: cn("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium", toneClasses[tone], className), children: [_jsx(Icon, { className: "h-3.5 w-3.5", "aria-hidden": "true" }), label] }));
}
export function MetricGrid({ id, children, columns = 4, className, }) {
    const columnsClass = {
        2: "md:grid-cols-2",
        3: "md:grid-cols-2 xl:grid-cols-3",
        4: "md:grid-cols-2 xl:grid-cols-4",
        5: "md:grid-cols-2 xl:grid-cols-5",
        6: "md:grid-cols-2 xl:grid-cols-6",
    }[columns];
    return _jsx("div", { id: id, className: cn("grid gap-3", columnsClass, className), children: children });
}
export function KpiCard({ label, value, detail, tone = "neutral", icon: Icon, footer, loading = false, className, }) {
    return (_jsxs("div", { className: cn("min-h-32 rounded-lg border border-border bg-card p-4 shadow-sm", className), children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsx("div", { className: "min-w-0 text-sm text-muted-foreground", children: label }), Icon ? (_jsx("span", { className: cn("rounded-md border p-1.5", toneClasses[tone]), children: _jsx(Icon, { className: "h-4 w-4", "aria-hidden": "true" }) })) : null] }), _jsx("div", { className: cn("mt-3 truncate text-3xl font-semibold text-foreground", loading && "h-9 animate-pulse rounded bg-muted text-transparent"), children: loading ? "loading" : value }), detail ? _jsx("div", { className: "mt-2 min-h-5 text-sm text-muted-foreground", children: detail }) : null, footer ? _jsx("div", { className: "mt-3 border-t border-border pt-3 text-xs text-muted-foreground", children: footer }) : null] }));
}
export function ProgressMetric({ label, value, max = 100, tone = "info", detail, }) {
    const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
    return (_jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex items-center justify-between gap-3 text-sm", children: [_jsx("span", { className: "text-muted-foreground", children: label }), _jsxs("span", { className: "font-medium text-foreground", children: [Math.round(pct), "%"] })] }), _jsx("div", { className: "h-2 overflow-hidden rounded-full bg-muted", children: _jsx("div", { className: cn("h-full rounded-full", toneClasses[tone].split(" ").find((part) => part.startsWith("bg-")) ?? "bg-primary"), style: { width: `${pct}%` } }) }), detail ? _jsx("div", { className: "text-xs text-muted-foreground", children: detail }) : null] }));
}
export function CapacityMeter({ used, limit, label = "Capacity", }) {
    const pct = limit > 0 ? used / limit : 0;
    const tone = pct >= 0.9 ? "critical" : pct >= 0.75 ? "warning" : "success";
    return (_jsx(ProgressMetric, { label: label, value: used, max: limit, tone: tone, detail: `${used.toLocaleString()} / ${limit.toLocaleString()}` }));
}
export function TrendDelta({ value, suffix = "%", }) {
    const tone = value > 0 ? "success" : value < 0 ? "critical" : "neutral";
    const sign = value > 0 ? "+" : "";
    return _jsxs(StatusPill, { tone: tone, children: [sign, value, suffix] });
}
//# sourceMappingURL=metrics.js.map