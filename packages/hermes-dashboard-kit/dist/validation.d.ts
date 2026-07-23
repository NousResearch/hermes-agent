import type { DashboardSnapshotContract } from "./contracts";
export type DashboardValidationSeverity = "error" | "warning";
export interface DashboardValidationIssue {
    severity: DashboardValidationSeverity;
    code: string;
    message: string;
    path: string;
}
export interface DashboardValidationResult {
    valid: boolean;
    issueCount: number;
    errorCount: number;
    warningCount: number;
    issues: DashboardValidationIssue[];
}
export interface DashboardAdoptionEntry {
    project: string;
    name: string;
    type: "package-native" | "static-adapter" | "custom" | string;
    targetPath?: string;
    status: "synced" | "partial" | "missing" | "package-native" | string;
    targetState: string;
    contractCoveragePercent?: number;
    workspaceCoveragePercent?: number;
    architectureEndpoint?: string;
}
export interface DashboardAdoptionRegistry {
    schemaVersion: number;
    source: {
        package: string;
        version: string;
        cssPath?: string;
        designContractPath?: string;
    };
    dashboards: DashboardAdoptionEntry[];
}
export declare function validateDashboardSnapshot(snapshot: DashboardSnapshotContract): DashboardValidationResult;
export declare function validateDashboardAdoptionRegistry(registry: DashboardAdoptionRegistry): DashboardValidationResult;
//# sourceMappingURL=validation.d.ts.map