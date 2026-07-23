import type { ReactNode } from "react";
export type DashboardOperationalStatus = "healthy" | "watch" | "degraded" | "blocked" | "unknown";
export type DashboardWorkspaceId = "command" | "operations" | "intelligence" | "capacity" | "projects" | "controls";
export type DashboardCapabilityId = "data-contracts" | "information-architecture" | "business-rules" | "implementation" | "design-system";
export interface DashboardDataSourceState {
    id: string;
    label: string;
    owner: string;
    endpoint?: string;
    freshnessSeconds?: number;
    lastUpdatedAt?: string;
    status: DashboardOperationalStatus;
    failureMode?: string;
}
export interface DashboardMetricContract {
    id: string;
    label: string;
    value: number | string;
    unit?: string;
    window?: "live" | "1h" | "4h" | "24h" | "7d" | "30d" | "90d";
    status?: DashboardOperationalStatus;
    detail?: ReactNode;
}
export interface DashboardActivityContract {
    id: string;
    title: string;
    happenedAt: string;
    actor?: string;
    projectId?: string;
    status: DashboardOperationalStatus;
    summary?: string;
}
export interface DashboardAlertContract {
    id: string;
    title: string;
    severity: Exclude<DashboardOperationalStatus, "healthy">;
    projectId?: string;
    owner?: string;
    openedAt: string;
    acknowledgedAt?: string;
    resolution?: string;
}
export interface DashboardDecisionContract {
    id: string;
    title: string;
    decidedAt: string;
    owner: string;
    decision: string;
    evidenceIds?: string[];
    revisitAt?: string;
}
export interface DashboardProjectStatusContract {
    id: string;
    name: string;
    domain: string;
    owner?: string;
    readinessPercent: number;
    status: DashboardOperationalStatus;
    summary: string;
    blockers?: DashboardAlertContract[];
    metrics?: DashboardMetricContract[];
}
export interface DashboardCostContract {
    id: string;
    provider: string;
    projectId?: string;
    costUsd?: number;
    usageUnit?: string;
    usageAmount?: number;
    budgetUsd?: number;
    window: "today" | "7d" | "30d" | "90d" | "month";
    status: DashboardOperationalStatus;
    lastUpdatedAt?: string;
}
export interface DashboardSystemHealthContract {
    id: string;
    projectId: string;
    status: DashboardOperationalStatus;
    uptimeSeconds?: number;
    cpuPercent?: number;
    memoryPercent?: number;
    queueDepth?: number;
    errorRatePercent?: number;
    lastCheckedAt?: string;
}
export interface DashboardReadinessSnapshotContract {
    id: string;
    projectId: string;
    readinessPercent: number;
    version?: string;
    status: DashboardOperationalStatus;
    generatedAt: string;
    openBlockers: number;
    completedMilestones: number;
    totalMilestones: number;
}
export interface DashboardModuleContract {
    id: string;
    label: string;
    workspace: DashboardWorkspaceId;
    route?: string;
    status: DashboardOperationalStatus;
    owner?: string;
    primaryQuestion: string;
    dataSources: DashboardDataSourceState[];
}
export interface DashboardSnapshotContract {
    id: string;
    projectId: string;
    generatedAt: string;
    modules: DashboardModuleContract[];
    metrics: DashboardMetricContract[];
    alerts: DashboardAlertContract[];
    activity: DashboardActivityContract[];
    decisions?: DashboardDecisionContract[];
    cost?: DashboardCostContract[];
    systemHealth?: DashboardSystemHealthContract[];
    readiness?: DashboardReadinessSnapshotContract;
}
export declare function clampReadinessPercent(value: number): number;
export declare function severityRank(status: DashboardOperationalStatus): number;
export declare function worstSeverity(statuses: DashboardOperationalStatus[]): DashboardOperationalStatus;
export declare function summarizeDashboardSnapshot(snapshot: DashboardSnapshotContract): {
    projectId: string;
    generatedAt: string;
    status: DashboardOperationalStatus;
    moduleCount: number;
    alertCount: number;
    criticalAlertCount: number;
    degradedSourceCount: number;
    readinessPercent: number | undefined;
};
//# sourceMappingURL=contracts.d.ts.map