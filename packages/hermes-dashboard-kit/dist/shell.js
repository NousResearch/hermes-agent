import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { cn } from "./utils";
export function DashboardShell({ sidebar, header, children, className, }) {
    return (_jsx("div", { className: cn("min-h-0 w-full", className), children: _jsxs("div", { className: "grid min-h-0 gap-4 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]", children: [sidebar ? _jsx("aside", { className: "min-w-0", children: sidebar }) : null, _jsxs("section", { className: "min-w-0 space-y-4", children: [header, _jsx(DashboardMain, { children: children })] })] }) }));
}
export function DashboardMain({ children, className, }) {
    return _jsx("main", { className: cn("min-w-0 space-y-4", className), children: children });
}
export function DashboardSection({ id, title, description, action, children, className, }) {
    return (_jsxs("section", { id: id, className: cn("rounded-lg border border-border bg-card p-4 shadow-sm", className), children: [(title || description || action) && (_jsxs("div", { className: "mb-4 flex flex-wrap items-start justify-between gap-3", children: [_jsxs("div", { className: "min-w-0", children: [title ? _jsx("h2", { className: "text-base font-semibold text-foreground", children: title }) : null, description ? _jsx("p", { className: "mt-1 text-sm text-muted-foreground", children: description }) : null] }), action] })), children] }));
}
export function DashboardHeader({ title, eyebrow, description, actions, meta, className, }) {
    return (_jsxs("header", { className: cn("flex flex-wrap items-start justify-between gap-4 rounded-lg border border-border bg-card p-4", className), children: [_jsxs("div", { className: "min-w-0", children: [eyebrow ? _jsx("div", { className: "mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground", children: eyebrow }) : null, _jsx("h1", { className: "truncate text-2xl font-semibold text-foreground", children: title }), description ? _jsx("p", { className: "mt-1 max-w-3xl text-sm text-muted-foreground", children: description }) : null, meta ? _jsx("div", { className: "mt-3 flex flex-wrap gap-2", children: meta }) : null] }), actions ? _jsx("div", { className: "flex shrink-0 flex-wrap items-center gap-2", children: actions }) : null] }));
}
export function DashboardSidebar({ title, description, items, footer, className, }) {
    return (_jsxs("nav", { className: cn("rounded-lg border border-border bg-card p-3", className), "aria-label": title, children: [_jsxs("div", { className: "border-b border-border pb-3", children: [_jsx("div", { className: "text-sm font-semibold text-foreground", children: title }), description ? _jsx("div", { className: "mt-1 text-xs text-muted-foreground", children: description }) : null] }), _jsx("div", { className: "mt-3 space-y-1", children: items.map((item) => {
                    const Icon = item.icon;
                    const content = (_jsxs(_Fragment, { children: [Icon ? _jsx(Icon, { className: "h-4 w-4 shrink-0", "aria-hidden": "true" }) : null, _jsx("span", { className: "min-w-0 flex-1 truncate", children: item.label }), item.badge] }));
                    const classes = cn("flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition", item.active
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground");
                    return item.href ? (_jsx("a", { className: classes, href: item.href, title: item.description ?? item.label, children: content }, item.id)) : (_jsx("button", { className: classes, onClick: item.onClick, title: item.description ?? item.label, type: "button", children: content }, item.id));
                }) }), footer ? _jsx("div", { className: "mt-4 border-t border-border pt-3", children: footer }) : null] }));
}
export function DashboardPageTitle({ children, className }) {
    return _jsx("h1", { className: cn("text-2xl font-semibold text-foreground", className), children: children });
}
//# sourceMappingURL=shell.js.map