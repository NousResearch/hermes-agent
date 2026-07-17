import type { ReactNode } from "react";
export declare function DashboardEmptyState({ title, description, action, className, }: {
    title?: string;
    description?: string;
    action?: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardLoadingState({ label, className, }: {
    label?: string;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardErrorState({ title, message, action, className, }: {
    title?: string;
    message?: string;
    action?: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=states.d.ts.map