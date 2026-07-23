import type { ReactNode } from "react";

export type DashboardOperationalStatus = "healthy" | "watch" | "degraded" | "blocked" | "unknown";

export type DashboardWorkspaceId =
  | "command"
  | "operations"
  | "intelligence"
  | "capacity"
  | "projects"
  | "controls";

export type DashboardCapabilityId =
  | "data-contracts"
  | "information-architecture"
  | "business-rules"
  | "implementation"
  | "design-system";

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

export function clampReadinessPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function severityRank(status: DashboardOperationalStatus): number {
  if (status === "blocked") return 4;
  if (status === "degraded") return 3;
  if (status === "watch") return 2;
  if (status === "unknown") return 1;
  return 0;
}

export function worstSeverity(statuses: DashboardOperationalStatus[]): DashboardOperationalStatus {
  return statuses.reduce<DashboardOperationalStatus>((worst, status) => (
    severityRank(status) > severityRank(worst) ? status : worst
  ), "healthy");
}

export function summarizeDashboardSnapshot(snapshot: DashboardSnapshotContract) {
  const sourceStatuses = snapshot.modules.flatMap((module) => module.dataSources.map((source) => source.status));
  const status = worstSeverity([
    snapshot.readiness?.status ?? "unknown",
    ...snapshot.modules.map((module) => module.status),
    ...snapshot.alerts.map((alert) => alert.severity),
    ...sourceStatuses,
  ]);

  return {
    projectId: snapshot.projectId,
    generatedAt: snapshot.generatedAt,
    status,
    moduleCount: snapshot.modules.length,
    alertCount: snapshot.alerts.length,
    criticalAlertCount: snapshot.alerts.filter((alert) => alert.severity === "blocked").length,
    degradedSourceCount: sourceStatuses.filter((sourceStatus) => severityRank(sourceStatus) >= severityRank("degraded")).length,
    readinessPercent: snapshot.readiness ? clampReadinessPercent(snapshot.readiness.readinessPercent) : undefined,
  };
}
