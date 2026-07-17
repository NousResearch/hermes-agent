import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Search } from "lucide-react";
import { cn } from "./utils";
import { DashboardEmptyState, DashboardErrorState, DashboardLoadingState } from "./states";
export function DataTable({ rows, columns, getRowKey, loading = false, error, emptyTitle, emptyDescription, onRowClick, className, }) {
    const [sort, setSort] = useState(null);
    const sortedRows = useMemo(() => {
        if (!sort)
            return rows;
        const column = columns.find((col) => col.id === sort.id);
        if (!column?.sortValue)
            return rows;
        return [...rows].sort((a, b) => {
            const av = column.sortValue?.(a);
            const bv = column.sortValue?.(b);
            const result = String(av ?? "").localeCompare(String(bv ?? ""), undefined, { numeric: true });
            return sort.dir === "asc" ? result : -result;
        });
    }, [columns, rows, sort]);
    if (loading)
        return _jsx(DashboardLoadingState, { label: "Loading table" });
    if (error)
        return _jsx(DashboardErrorState, { message: error });
    if (!rows.length)
        return _jsx(DashboardEmptyState, { title: emptyTitle, description: emptyDescription });
    return (_jsx("div", { className: cn("overflow-hidden rounded-lg border border-border bg-card", className), children: _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "w-full min-w-[42rem] border-collapse text-sm", children: [_jsx("thead", { className: "bg-muted/60 text-left text-xs uppercase tracking-wide text-muted-foreground", children: _jsx("tr", { children: columns.map((column) => {
                                const active = sort?.id === column.id;
                                const Icon = active && sort?.dir === "desc" ? ChevronDown : ChevronUp;
                                return (_jsx("th", { className: cn("border-b border-border px-3 py-2 font-medium", column.className), children: column.sortValue ? (_jsxs("button", { className: "inline-flex items-center gap-1 text-left", onClick: () => setSort((current) => current?.id === column.id && current.dir === "asc" ? { id: column.id, dir: "desc" } : { id: column.id, dir: "asc" }), type: "button", children: [column.header, _jsx(Icon, { className: cn("h-3.5 w-3.5", active ? "opacity-100" : "opacity-30"), "aria-hidden": "true" })] })) : (column.header) }, column.id));
                            }) }) }), _jsx("tbody", { children: sortedRows.map((row, index) => (_jsx("tr", { className: cn("border-b border-border last:border-b-0", onRowClick && "cursor-pointer hover:bg-muted/50"), onClick: () => onRowClick?.(row), children: columns.map((column) => (_jsx("td", { className: cn("px-3 py-2 align-top", column.cellClassName), children: column.accessor ? column.accessor(row) : null }, column.id))) }, getRowKey(row, index)))) })] }) }) }));
}
export function TableToolbar({ children, className, }) {
    return _jsx("div", { className: cn("flex flex-wrap items-center justify-between gap-3", className), children: children });
}
export function FilterBar({ children, className, }) {
    return _jsx("div", { className: cn("flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card p-2", className), children: children });
}
export function SearchInput({ value, onChange, placeholder = "Search", className, }) {
    return (_jsxs("label", { className: cn("relative block min-w-56 flex-1", className), children: [_jsx(Search, { className: "pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground", "aria-hidden": "true" }), _jsx("input", { className: "h-9 w-full rounded-md border border-border bg-background px-9 text-sm text-foreground outline-none transition focus:border-primary", onChange: (event) => onChange(event.target.value), placeholder: placeholder, type: "search", value: value })] }));
}
export function SegmentedControl({ value, options, onChange, className, }) {
    return (_jsx("div", { className: cn("inline-flex rounded-md border border-border bg-muted p-1", className), children: options.map((option) => (_jsx("button", { className: cn("h-7 rounded px-3 text-xs font-medium transition", option.value === value ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"), onClick: () => onChange(option.value), type: "button", children: option.label }, option.value))) }));
}
export function DateRangeToggle({ value, onChange, }) {
    return (_jsx(SegmentedControl, { value: value, onChange: onChange, options: [
            { value: "7d", label: "7 days" },
            { value: "30d", label: "30 days" },
            { value: "90d", label: "90 days" },
        ] }));
}
//# sourceMappingURL=data-table.js.map