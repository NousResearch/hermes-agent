import type { DashboardModuleContract, DashboardWorkspaceId } from "./contracts";
export interface DashboardWorkspaceDefinition {
    id: DashboardWorkspaceId;
    label: string;
    primaryQuestion: string;
    description: string;
    requiredCapabilities: string[];
}
export declare const HERMES_DASHBOARD_WORKSPACES: DashboardWorkspaceDefinition[];
export declare function getDashboardWorkspace(id: DashboardWorkspaceId): DashboardWorkspaceDefinition;
export declare function groupDashboardModulesByWorkspace(modules: DashboardModuleContract[]): {
    modules: DashboardModuleContract[];
    id: DashboardWorkspaceId;
    label: string;
    primaryQuestion: string;
    description: string;
    requiredCapabilities: string[];
}[];
//# sourceMappingURL=workspaces.d.ts.map