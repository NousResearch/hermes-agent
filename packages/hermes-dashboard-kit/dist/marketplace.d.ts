export type DashboardPluginSurface = "page" | "panel" | "command" | "signal" | "theme";
export type DashboardPluginPermission = "viewer" | "operator" | "admin";
export interface DashboardPluginCommand {
    id: string;
    label: string;
    permission: DashboardPluginPermission;
    riskLevel: "low" | "medium" | "high";
    description?: string;
}
export interface DashboardPluginPanel {
    id: string;
    label: string;
    surface: DashboardPluginSurface;
    route?: string;
    signalContract?: string;
}
export interface DashboardPluginManifest {
    id: string;
    label: string;
    owner: string;
    category: string;
    version: string;
    description: string;
    healthUrl?: string;
    productionUrl?: string;
    localUrl?: string;
    panels: DashboardPluginPanel[];
    commands: DashboardPluginCommand[];
    signals: string[];
    permissions: DashboardPluginPermission[];
}
export declare function dashboardPluginHasSignal(plugin: DashboardPluginManifest, signal: string): boolean;
export declare function dashboardPluginRequiresAdmin(plugin: DashboardPluginManifest): boolean;
//# sourceMappingURL=marketplace.d.ts.map