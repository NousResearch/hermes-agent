import type { ReactNode } from "react";
import { type DashboardTone } from "./metrics";
export interface ExecutiveProjectScore {
    id: string;
    name: string;
    owner?: string;
    domain: string;
    status: string;
    tone?: DashboardTone;
    healthScore?: number;
    summary: string;
    metrics?: Array<{
        label: string;
        value: ReactNode;
    }>;
}
export interface ExecutiveActionItem {
    id: string;
    title: string;
    owner: string;
    urgency: "low" | "normal" | "high" | "critical";
    due?: string;
    source?: string;
}
export interface ExecutiveRollupMetric {
    label: string;
    value: ReactNode;
    detail?: ReactNode;
    tone?: DashboardTone;
}
export interface ExecutiveDomainTab {
    id: string;
    label: string;
    status?: string;
    tone?: DashboardTone;
}
export declare function ExecutiveHealthRollup({ metrics, className, }: {
    metrics: ExecutiveRollupMetric[];
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ExecutiveProjectScorecard({ project, className, }: {
    project: ExecutiveProjectScore;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ExecutiveActionQueue({ id, items, title, className, }: {
    id?: string;
    items: ExecutiveActionItem[];
    title?: string;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ExecutiveCostCapacityRollup({ cost, capacity, throughput, className, }: {
    cost: ExecutiveRollupMetric;
    capacity: ExecutiveRollupMetric;
    throughput: ExecutiveRollupMetric;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ExecutiveDomainTabs({ tabs, activeId, onSelect, className, }: {
    tabs: ExecutiveDomainTab[];
    activeId: string;
    onSelect?: (id: string) => void;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=executive.d.ts.map