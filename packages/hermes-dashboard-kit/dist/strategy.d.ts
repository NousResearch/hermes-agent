import type { DashboardCapabilityId, DashboardOperationalStatus, DashboardSnapshotContract } from "./contracts";
export interface DashboardCapabilityAssessment {
    id: DashboardCapabilityId;
    label: string;
    status: DashboardOperationalStatus;
    gap: string;
    nextAction: string;
}
export interface DashboardArchitectureAssessment {
    projectId: string;
    status: DashboardOperationalStatus;
    readinessPercent?: number;
    workspaceCoveragePercent: number;
    capabilities: DashboardCapabilityAssessment[];
    recommendedNextActions: string[];
}
export declare function assessDashboardArchitecture(snapshot: DashboardSnapshotContract): DashboardArchitectureAssessment;
//# sourceMappingURL=strategy.d.ts.map