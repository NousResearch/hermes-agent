import type { DashboardTone } from "./metrics";
export type DashboardFreshness = "fresh" | "aging" | "stale" | "unknown";
export type DashboardHealthCondition = "healthy" | "degraded" | "critical" | "unknown";
export type DashboardQueueState = "queued" | "running" | "completed" | "failed" | "blocked" | "stale";
export type DashboardSeverity = "low" | "normal" | "high" | "critical";
export interface DashboardSignalSource {
    id: string;
    label: string;
    owner: string;
    category: string;
    projectName?: string;
    projectPath?: string;
    url?: string;
    healthUrl?: string;
}
export interface HealthSnapshot {
    state: DashboardHealthCondition;
    score?: number;
    message?: string;
    checkedAt?: string;
    freshness?: DashboardFreshness;
}
export interface CostSnapshot {
    period: "24h" | "7d" | "30d" | "90d" | "unknown";
    known: boolean;
    amountUsd?: number;
    tokenCount?: number;
    apiCalls?: number;
    storageBytes?: number;
    message?: string;
}
export interface CapacitySnapshot {
    known: boolean;
    used?: number;
    limit?: number;
    floor?: number;
    ceiling?: number;
    pressure?: "low" | "medium" | "high" | "unknown";
    message?: string;
}
export interface QueueSnapshot {
    queued: number;
    running: number;
    failed: number;
    blocked: number;
    stale: number;
    completed?: number;
}
export interface ActionNeeded {
    id: string;
    title: string;
    owner: string;
    severity: DashboardSeverity;
    sourceDashboardId: string;
    source?: string;
    due?: string;
    nextStep?: string;
}
export interface ResearchSignal {
    findings?: number;
    evidence?: number;
    confidence?: number;
    staleFindings?: number;
    message?: string;
}
export interface DeploymentSignal {
    environment: "local" | "staging" | "production" | "unknown";
    status: "current" | "pending" | "failed" | "unknown";
    version?: string;
    commit?: string;
    deployedAt?: string;
    message?: string;
}
export interface DashboardSnapshot {
    source: DashboardSignalSource;
    health: HealthSnapshot;
    cost?: CostSnapshot;
    capacity?: CapacitySnapshot;
    queue?: QueueSnapshot;
    actions?: ActionNeeded[];
    research?: ResearchSignal;
    deployment?: DeploymentSignal;
    updatedAt?: string;
}
export declare function dashboardToneForHealth(state: DashboardHealthCondition): DashboardTone;
export declare function dashboardToneForSeverity(severity: DashboardSeverity): DashboardTone;
export declare function dashboardFreshness(checkedAt?: string, now?: number): DashboardFreshness;
export declare function dashboardHealthScore(snapshot: DashboardSnapshot): number;
//# sourceMappingURL=signals.d.ts.map