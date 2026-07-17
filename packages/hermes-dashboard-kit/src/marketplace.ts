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

export function dashboardPluginHasSignal(plugin: DashboardPluginManifest, signal: string) {
  return plugin.signals.includes(signal);
}

export function dashboardPluginRequiresAdmin(plugin: DashboardPluginManifest) {
  return plugin.commands.some((command) => command.permission === "admin" || command.riskLevel === "high");
}
