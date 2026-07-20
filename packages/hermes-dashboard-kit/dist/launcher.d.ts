import { type ReactNode } from "react";
export type DashboardHealthStatus = "current" | "online" | "offline" | "stale" | "checking" | "unknown" | "missing";
export interface DashboardRegistryEntry {
    id: string;
    label: string;
    description?: string;
    url?: string;
    localUrl?: string;
    productionUrl?: string;
    healthUrl?: string;
    snapshotUrl?: string;
    status?: DashboardHealthStatus;
    category?: string;
    owner?: string;
}
export interface DashboardHealthState {
    status: DashboardHealthStatus;
    checkedAt?: string;
    message?: string;
}
export declare function useDashboardHealth(dashboards: DashboardRegistryEntry[], { enabled, intervalMs, }?: {
    enabled?: boolean;
    intervalMs?: number;
}): Record<string, DashboardHealthState>;
export declare function DashboardLauncher({ dashboards, currentId, title, empty, className, pollHealth, healthPollIntervalMs, }: {
    dashboards: DashboardRegistryEntry[];
    currentId?: string;
    title?: string;
    empty?: ReactNode;
    className?: string;
    pollHealth?: boolean;
    healthPollIntervalMs?: number;
}): import("react/jsx-runtime").JSX.Element;
export declare function ProjectSwitcher({ projects, currentId, onChange, label, }: {
    projects: {
        id: string;
        label: string;
        status?: DashboardHealthStatus;
    }[];
    currentId?: string;
    onChange?: (id: string) => void;
    label?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=launcher.d.ts.map