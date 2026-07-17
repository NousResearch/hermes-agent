import type { ComponentType, ReactNode } from "react";
export type DashboardTone = "neutral" | "success" | "warning" | "critical" | "info" | "unknown";
export declare function StatusPill({ children, tone, className, }: {
    children: ReactNode;
    tone?: DashboardTone;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function HealthBadge({ label, tone, className, }: {
    label: string;
    tone?: DashboardTone;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function MetricGrid({ id, children, columns, className, }: {
    id?: string;
    children: ReactNode;
    columns?: 2 | 3 | 4 | 5 | 6;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function KpiCard({ label, value, detail, tone, icon: Icon, footer, loading, className, }: {
    label: string;
    value: ReactNode;
    detail?: ReactNode;
    tone?: DashboardTone;
    icon?: ComponentType<{
        className?: string;
    }>;
    footer?: ReactNode;
    loading?: boolean;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ProgressMetric({ label, value, max, tone, detail, }: {
    label: string;
    value: number;
    max?: number;
    tone?: DashboardTone;
    detail?: ReactNode;
}): import("react/jsx-runtime").JSX.Element;
export declare function CapacityMeter({ used, limit, label, }: {
    used: number;
    limit: number;
    label?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function TrendDelta({ value, suffix, }: {
    value: number;
    suffix?: string;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=metrics.d.ts.map