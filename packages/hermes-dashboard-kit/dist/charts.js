import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Fragment } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, } from "recharts";
import { cn } from "./utils";
import { DashboardEmptyState } from "./states";
import { StatusPill } from "./metrics";
export function ChartPanel({ id, title, description, action, children, className, }) {
    return (_jsxs("section", { id: id, className: cn("rounded-lg border border-border bg-card p-4 shadow-sm", className), children: [_jsxs("div", { className: "mb-4 flex flex-wrap items-start justify-between gap-3", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-base font-semibold text-foreground", children: title }), description ? _jsx("p", { className: "mt-1 text-sm text-muted-foreground", children: description }) : null] }), action] }), children] }));
}
export function SimpleBarChart({ data, valueLabel = "value", }) {
    if (!data.length)
        return _jsx(DashboardEmptyState, { title: "No chart data", description: "No series values are available." });
    return (_jsx("div", { className: "h-56 min-w-0 rounded-md border border-border bg-background p-3", role: "img", "aria-label": `bar chart by ${valueLabel}`, children: _jsx(ResponsiveContainer, { height: "100%", width: "100%", children: _jsxs(BarChart, { data: data, margin: { bottom: 0, left: -18, right: 8, top: 8 }, children: [_jsx(CartesianGrid, { stroke: "hsl(var(--border))", strokeDasharray: "3 3", vertical: false }), _jsx(XAxis, { dataKey: "label", fontSize: 12, stroke: "hsl(var(--muted-foreground))", tickLine: false }), _jsx(YAxis, { fontSize: 12, stroke: "hsl(var(--muted-foreground))", tickLine: false }), _jsx(Tooltip, { contentStyle: {
                            background: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                            color: "hsl(var(--foreground))",
                        }, labelStyle: { color: "hsl(var(--foreground))" } }), _jsx(Bar, { dataKey: "value", fill: "hsl(var(--primary))", name: valueLabel, radius: [6, 6, 0, 0] })] }) }) }));
}
export function SimpleLineChart({ data, height = 180, }) {
    if (data.length < 2)
        return _jsx(DashboardEmptyState, { title: "Not enough data", description: "At least two points are required." });
    return (_jsx("div", { className: "min-w-0 rounded-md border border-border bg-background p-3", style: { height }, role: "img", "aria-label": "line chart", children: _jsx(ResponsiveContainer, { height: "100%", width: "100%", children: _jsxs(LineChart, { data: data, margin: { bottom: 0, left: -18, right: 8, top: 8 }, children: [_jsx(CartesianGrid, { stroke: "hsl(var(--border))", strokeDasharray: "3 3", vertical: false }), _jsx(XAxis, { dataKey: "label", fontSize: 12, stroke: "hsl(var(--muted-foreground))", tickLine: false }), _jsx(YAxis, { fontSize: 12, stroke: "hsl(var(--muted-foreground))", tickLine: false }), _jsx(Tooltip, { contentStyle: {
                            background: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                            color: "hsl(var(--foreground))",
                        }, labelStyle: { color: "hsl(var(--foreground))" } }), _jsx(Line, { activeDot: { r: 5 }, dataKey: "value", dot: { r: 3 }, stroke: "hsl(var(--primary))", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 3, type: "monotone" })] }) }) }));
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