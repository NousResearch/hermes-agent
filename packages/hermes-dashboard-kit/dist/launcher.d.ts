import type { ReactNode } from "react";
export interface DashboardRegistryEntry {
    id: string;
    label: string;
    description?: string;
    url?: string;
    localUrl?: string;
    productionUrl?: string;
    healthUrl?: string;
    status?: "current" | "online" | "offline" | "unknown" | "missing";
    category?: string;
    owner?: string;
}
export declare function DashboardLauncher({ dashboards, currentId, title, empty, className, }: {
    dashboards: DashboardRegistryEntry[];
    currentId?: string;
    title?: string;
    empty?: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ProjectSwitcher({ projects, currentId, onChange, label, }: {
    projects: {
        id: string;
        label: string;
        status?: DashboardRegistryEntry["status"];
    }[];
    currentId?: string;
    onChange?: (id: string) => void;
    label?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=launcher.d.ts.map