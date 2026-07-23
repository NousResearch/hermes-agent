import type { DashboardCapabilityId, DashboardOperationalStatus, DashboardSnapshotContract } from "./contracts";
import { severityRank, summarizeDashboardSnapshot } from "./contracts";
import { HERMES_DASHBOARD_WORKSPACES } from "./workspaces";

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

function capabilityStatus(condition: boolean, warningCondition = false): DashboardOperationalStatus {
  if (condition) return "healthy";
  if (warningCondition) return "watch";
  return "degraded";
}

export function assessDashboardArchitecture(snapshot: DashboardSnapshotContract): DashboardArchitectureAssessment {
  const summary = summarizeDashboardSnapshot(snapshot);
  const coveredWorkspaces = new Set(snapshot.modules.map((module) => module.workspace));
  const workspaceCoveragePercent = Math.round((coveredWorkspaces.size / HERMES_DASHBOARD_WORKSPACES.length) * 100);
  const hasSources = snapshot.modules.some((module) => module.dataSources.length > 0);
  const hasRulesSignals = snapshot.alerts.length > 0 || snapshot.decisions?.length;
  const hasImplementationSignals = snapshot.modules.some((module) => module.route);
  const usesArchitecture = workspaceCoveragePercent >= 80;

  const capabilities: DashboardCapabilityAssessment[] = [
    {
      id: "data-contracts",
      label: "Data contracts",
      status: capabilityStatus(hasSources, snapshot.metrics.length > 0),
      gap: hasSources ? "Data sources are declared." : "Dashboard modules do not expose owned data sources and freshness.",
      nextAction: "Attach endpoint, owner, freshness, and failure-mode metadata to every dashboard module.",
    },
    {
      id: "information-architecture",
      label: "Information architecture",
      status: capabilityStatus(usesArchitecture, workspaceCoveragePercent >= 50),
      gap: usesArchitecture ? "Workspace coverage is broad enough for an operator surface." : "Dashboard modules are not yet grouped around the six Hermes workspaces.",
      nextAction: "Collapse project-specific tabs into Command, Operations, Intelligence, Capacity, Projects, and Controls.",
    },
    {
      id: "business-rules",
      label: "Business rules",
      status: capabilityStatus(Boolean(hasRulesSignals), snapshot.metrics.length > 0),
      gap: hasRulesSignals ? "Judgment signals exist." : "The dashboard has metrics, but few explicit rules for healthy, watch, degraded, or blocked.",
      nextAction: "Define per-project rules that turn raw metrics into alerts, decisions, and recommendations.",
    },
    {
      id: "implementation",
      label: "Production implementation",
      status: capabilityStatus(hasImplementationSignals, snapshot.modules.length > 0),
      gap: hasImplementationSignals ? "At least one module has a route." : "Modules are modeled but not tied to production routes.",
      nextAction: "Connect modules to real routes, tests, health checks, and deployment evidence.",
    },
    {
      id: "design-system",
      label: "Shared design system",
      status: capabilityStatus(summary.degradedSourceCount === 0 && usesArchitecture, summary.degradedSourceCount === 0),
      gap: summary.degradedSourceCount === 0 ? "No degraded data sources are currently reported." : "Data-source failures will make even well-designed dashboards untrustworthy.",
      nextAction: "Use dashboard-kit primitives and record any local exceptions in the adoption registry.",
    },
  ];

  const status = capabilities.reduce<DashboardOperationalStatus>((worst, capability) => (
    severityRank(capability.status) > severityRank(worst) ? capability.status : worst
  ), summary.status);

  return {
    projectId: snapshot.projectId,
    status,
    readinessPercent: summary.readinessPercent,
    workspaceCoveragePercent,
    capabilities,
    recommendedNextActions: capabilities
      .filter((capability) => capability.status !== "healthy")
      .sort((a, b) => severityRank(b.status) - severityRank(a.status))
      .map((capability) => capability.nextAction),
  };
}
