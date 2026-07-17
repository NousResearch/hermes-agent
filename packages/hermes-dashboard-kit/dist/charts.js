import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Fragment } from "react";
import { cn } from "./utils";
import { DashboardEmptyState } from "./states";
import { StatusPill } from "./metrics";
export function ChartPanel({ id, title, description, action, children, className, }) {
    return (_jsxs("section", { id: id, className: cn("rounded-lg border border-border bg-card p-4 shadow-sm", className), children: [_jsxs("div", { className: "mb-4 flex flex-wrap items-start justify-between gap-3", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-base font-semibold text-foreground", children: title }), description ? _jsx("p", { className: "mt-1 text-sm text-muted-foreground", children: description }) : null] }), action] }), children] }));
}
export function SimpleBarChart({ data, valueLabel = "value", }) {
    if (!data.length)
        return _jsx(DashboardEmptyState, { title: "No chart data", description: "No series values are available." });
    const max = Math.max(...data.map((item) => item.value), 1);
    return (_jsx("div", { className: "space-y-3", children: data.map((item) => (_jsxs("div", { className: "grid grid-cols-[minmax(5rem,10rem)_minmax(0,1fr)_4rem] items-center gap-3 text-sm", children: [_jsx("div", { className: "truncate text-muted-foreground", children: item.label }), _jsx("div", { className: "h-3 overflow-hidden rounded-full bg-muted", children: _jsx("div", { "aria-label": `${item.label} ${valueLabel}: ${item.value}`, className: "h-full rounded-full bg-primary", style: { width: `${Math.max(2, (item.value / max) * 100)}%` } }) }), _jsx("div", { className: "text-right font-mono-ui text-xs text-foreground", children: item.value })] }, item.label))) }));
}
export function SimpleLineChart({ data, height = 140, }) {
    if (data.length < 2)
        return _jsx(DashboardEmptyState, { title: "Not enough data", description: "At least two points are required." });
    const width = 640;
    const min = Math.min(...data.map((item) => item.value));
    const max = Math.max(...data.map((item) => item.value));
    const span = max - min || 1;
    const points = data.map((item, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((item.value - min) / span) * (height - 12) - 6;
        return `${x},${y}`;
    }).join(" ");
    return (_jsx("div", { className: "overflow-hidden rounded-md border border-border bg-background p-3", children: _jsx("svg", { "aria-label": "line chart", className: "h-auto w-full", viewBox: `0 0 ${width} ${height}`, role: "img", children: _jsx("polyline", { fill: "none", points: points, stroke: "hsl(var(--primary))", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "3" }) }) }));
}
export function HeatmapGrid({ rows, columns, values, }) {
    const max = Math.max(...Object.values(values), 1);
    return (_jsx("div", { className: "overflow-x-auto", children: _jsxs("div", { className: "grid min-w-[36rem] gap-1", style: { gridTemplateColumns: `8rem repeat(${columns.length}, minmax(5rem, 1fr))` }, children: [_jsx("div", {}), columns.map((column) => _jsx("div", { className: "px-2 py-1 text-xs font-medium text-muted-foreground", children: column }, column)), rows.map((row) => (_jsxs(Fragment, { children: [_jsx("div", { className: "px-2 py-2 text-sm text-muted-foreground", children: row }, `${row}-label`), columns.map((column) => {
                            const key = `${row}:${column}`;
                            const value = values[key] ?? 0;
                            const opacity = value > 0 ? 0.15 + (value / max) * 0.55 : 0.06;
                            return (_jsx("div", { className: "rounded-md border border-border px-2 py-2 text-center font-mono-ui text-sm text-foreground", style: { backgroundColor: `rgb(20 184 166 / ${opacity})` }, children: value }, key));
                        })] }, row)))] }) }));
}
export function InsightPanel({ title, children, tone = "info", }) {
    return (_jsxs("section", { className: "rounded-lg border border-border bg-card p-4", children: [_jsxs("div", { className: "mb-3 flex items-center justify-between gap-3", children: [_jsx("h2", { className: "text-base font-semibold text-foreground", children: title }), _jsx(StatusPill, { tone: tone, children: tone })] }), _jsx("div", { className: "space-y-3", children: children })] }));
}
export function FindingCard({ title, description, evidence, tone = "info", }) {
    return (_jsxs("article", { className: "rounded-md border border-border bg-background p-3", children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsx("div", { className: "font-medium text-foreground", children: title }), _jsx(StatusPill, { tone: tone, children: tone })] }), description ? _jsx("p", { className: "mt-2 text-sm text-muted-foreground", children: description }) : null, evidence ? _jsx("div", { className: "mt-3 text-xs text-muted-foreground", children: evidence }) : null] }));
}
export function RecommendationCard({ title, action, confidence, }) {
    return (_jsxs("article", { className: "rounded-md border border-border bg-background p-3", children: [_jsx("div", { className: "font-medium text-foreground", children: title }), _jsx("p", { className: "mt-2 text-sm text-muted-foreground", children: action }), typeof confidence === "number" ? (_jsxs("div", { className: "mt-3 text-xs text-muted-foreground", children: ["Confidence: ", Math.round(confidence * 100), "%"] })) : null] }));
}
//# sourceMappingURL=charts.js.map