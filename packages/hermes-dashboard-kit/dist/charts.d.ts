import { type ReactNode } from "react";
import { type DashboardTone } from "./metrics";
export declare function ChartPanel({ id, title, description, action, children, className, }: {
    id?: string;
    title: string;
    description?: string;
    action?: ReactNode;
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function SimpleBarChart({ data, valueLabel, }: {
    data: {
        label: string;
        value: number;
    }[];
    valueLabel?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function SimpleLineChart({ data, height, }: {
    data: {
        label: string;
        value: number;
    }[];
    height?: number;
}): import("react/jsx-runtime").JSX.Element;
export declare function HeatmapGrid({ rows, columns, values, }: {
    rows: string[];
    columns: string[];
    values: Record<string, number>;
}): import("react/jsx-runtime").JSX.Element;
export declare function InsightPanel({ title, children, tone, }: {
    title: string;
    children: ReactNode;
    tone?: DashboardTone;
}): import("react/jsx-runtime").JSX.Element;
export declare function FindingCard({ title, description, evidence, tone, }: {
    title: string;
    description?: string;
    evidence?: ReactNode;
    tone?: DashboardTone;
}): import("react/jsx-runtime").JSX.Element;
export declare function RecommendationCard({ title, action, confidence, }: {
    title: string;
    action: string;
    confidence?: number;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=charts.d.ts.map