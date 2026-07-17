import type { ComponentType, ReactNode } from "react";
export interface DashboardNavItem {
    id: string;
    label: string;
    href?: string;
    active?: boolean;
    icon?: ComponentType<{
        className?: string;
    }>;
    badge?: ReactNode;
    description?: string;
    onClick?: () => void;
}
export declare function DashboardShell({ sidebar, header, children, className, }: {
    sidebar?: ReactNode;
    header?: ReactNode;
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardMain({ children, className, }: {
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardSection({ id, title, description, action, children, className, }: {
    id?: string;
    title?: string;
    description?: string;
    action?: ReactNode;
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardHeader({ title, eyebrow, description, actions, meta, className, }: {
    title: string;
    eyebrow?: string;
    description?: string;
    actions?: ReactNode;
    meta?: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardSidebar({ title, description, items, footer, className, }: {
    title: string;
    description?: string;
    items: DashboardNavItem[];
    footer?: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DashboardPageTitle({ children, className }: {
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=shell.d.ts.map