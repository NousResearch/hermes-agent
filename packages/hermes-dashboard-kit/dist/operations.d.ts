import type { ComponentType } from "react";
import { type DashboardTone } from "./metrics";
export interface CommandAction {
    id: string;
    label: string;
    description?: string;
    icon?: ComponentType<{
        className?: string;
    }>;
    tone?: DashboardTone;
    disabled?: boolean;
    disabledReason?: string;
    permission?: "viewer" | "operator" | "admin";
    requiresConfirmation?: boolean;
    riskLevel?: "low" | "medium" | "high";
    onClick?: () => void;
}
export declare function CommandBar({ title, description, actions, className, }: {
    title?: string;
    description?: string;
    actions: CommandAction[];
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function ActionButtonGroup({ actions }: {
    actions: CommandAction[];
}): import("react/jsx-runtime").JSX.Element;
export interface ActivityEvent {
    id: string;
    title: string;
    timestamp?: string;
    description?: string;
    tone?: DashboardTone;
}
export declare function ActivityTimeline({ events, empty, }: {
    events: ActivityEvent[];
    empty?: string;
}): import("react/jsx-runtime").JSX.Element;
export type QueueStatus = "queued" | "running" | "completed" | "failed" | "blocked" | "stale";
export interface QueueItem {
    id: string;
    label: string;
    status: QueueStatus;
    detail?: string;
}
export declare function QueuePanel({ title, items, }: {
    title?: string;
    items: QueueItem[];
}): import("react/jsx-runtime").JSX.Element;
export declare function RunStatusPanel({ running, queued, failed, completed, }: {
    running: number;
    queued: number;
    failed: number;
    completed: number;
}): import("react/jsx-runtime").JSX.Element;
export declare function AuditEventList({ events }: {
    events: ActivityEvent[];
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=operations.d.ts.map