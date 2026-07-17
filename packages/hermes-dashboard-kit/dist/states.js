import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { cn } from "./utils";
export function DashboardEmptyState({ title = "No data", description = "There is nothing to show yet.", action, className, }) {
    return (_jsxs("div", { className: cn("flex min-h-32 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card/50 p-6 text-center", className), children: [_jsx(Inbox, { className: "h-5 w-5 text-muted-foreground", "aria-hidden": "true" }), _jsxs("div", { children: [_jsx("div", { className: "text-sm font-medium text-foreground", children: title }), _jsx("div", { className: "mt-1 text-sm text-muted-foreground", children: description })] }), action] }));
}
export function DashboardLoadingState({ label = "Loading", className, }) {
    return (_jsxs("div", { className: cn("flex min-h-32 items-center justify-center gap-2 rounded-lg border border-border bg-card/60 p-6 text-sm text-muted-foreground", className), children: [_jsx(Loader2, { className: "h-4 w-4 animate-spin", "aria-hidden": "true" }), _jsx("span", { children: label })] }));
}
export function DashboardErrorState({ title = "Unable to load", message, action, className, }) {
    return (_jsxs("div", { className: cn("flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm", className), role: "alert", children: [_jsx(AlertTriangle, { className: "mt-0.5 h-4 w-4 shrink-0 text-destructive", "aria-hidden": "true" }), _jsxs("div", { className: "min-w-0 flex-1", children: [_jsx("div", { className: "font-medium text-foreground", children: title }), message ? _jsx("div", { className: "mt-1 text-muted-foreground", children: message }) : null, action ? _jsx("div", { className: "mt-3", children: action }) : null] })] }));
}
//# sourceMappingURL=states.js.map