import type { DashboardModuleContract, DashboardWorkspaceId } from "./contracts";

export interface DashboardWorkspaceDefinition {
  id: DashboardWorkspaceId;
  label: string;
  primaryQuestion: string;
  description: string;
  requiredCapabilities: string[];
}

export const HERMES_DASHBOARD_WORKSPACES: DashboardWorkspaceDefinition[] = [
  {
    id: "command",
    label: "Command",
    primaryQuestion: "What needs attention now?",
    description: "The daily operating surface for blockers, active risks, current status, and priority commands.",
    requiredCapabilities: ["alert rollup", "current status", "operator command queue", "decision log"],
  },
  {
    id: "operations",
    label: "Operations",
    primaryQuestion: "What is running, blocked, stale, expensive, or failing?",
    description: "Live execution, queues, schedules, deployment state, health, errors, and production diagnostics.",
    requiredCapabilities: ["run monitor", "queue health", "system health", "deployment evidence"],
  },
  {
    id: "intelligence",
    label: "Intelligence",
    primaryQuestion: "What have we learned?",
    description: "Evidence, findings, research reports, market or audience learning, confidence, and recommendations.",
    requiredCapabilities: ["findings", "evidence", "confidence", "research status"],
  },
  {
    id: "capacity",
    label: "Capacity",
    primaryQuestion: "What are we spending, consuming, scanning, generating, or storing?",
    description: "Cost, token use, API calls, storage growth, scan throughput, and budget posture.",
    requiredCapabilities: ["budget", "usage", "throughput", "storage"],
  },
  {
    id: "projects",
    label: "Projects",
    primaryQuestion: "How is each business unit doing?",
    description: "Cross-project readiness, ownership, blockers, production coverage, and promotion posture.",
    requiredCapabilities: ["readiness", "ownership", "blockers", "fleet health"],
  },
  {
    id: "controls",
    label: "Controls",
    primaryQuestion: "What can I start, stop, approve, tune, or deploy?",
    description: "Permission-aware controls for safe operations, approvals, capacity changes, and deployment actions.",
    requiredCapabilities: ["permissions", "confirmation gates", "audit trail", "rollback evidence"],
  },
];

export function getDashboardWorkspace(id: DashboardWorkspaceId): DashboardWorkspaceDefinition {
  const workspace = HERMES_DASHBOARD_WORKSPACES.find((item) => item.id === id);
  if (!workspace) {
    throw new Error(`Unknown dashboard workspace: ${id}`);
  }
  return workspace;
}

export function groupDashboardModulesByWorkspace(modules: DashboardModuleContract[]) {
  return HERMES_DASHBOARD_WORKSPACES.map((workspace) => ({
    ...workspace,
    modules: modules.filter((module) => module.workspace === workspace.id),
  }));
}
