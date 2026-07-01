export type RiskLevel = "read_only" | "disabled" | "critical";

export type StatusCard = {
  label: string;
  value: string;
  tone: "ok" | "warn" | "muted";
};

export type QuickAction = {
  label: string;
  command: string;
  risk: RiskLevel;
  description: string;
};

export type LogLine = {
  level: "info" | "warn" | "error";
  message: string;
  time: string;
};

export const statusCards: StatusCard[] = [
  { label: "Gateway", value: "Online", tone: "ok" },
  { label: "Mode", value: "Read-only MVP", tone: "muted" },
  { label: "Actions", value: "Disabled", tone: "warn" },
  { label: "Approvals", value: "0 pending", tone: "ok" },
];

export const quickActions: QuickAction[] = [
  {
    label: "Refresh status",
    command: "mock:refresh-status",
    risk: "read_only",
    description: "Local UI-only refresh. No Hermes API call yet.",
  },
  {
    label: "Open sessions",
    command: "mock:open-sessions",
    risk: "read_only",
    description: "Preview the sessions panel with mock data.",
  },
  {
    label: "Run healthcheck",
    command: "disabled:healthcheck",
    risk: "disabled",
    description: "Requires sidecar + approve gate in a later milestone.",
  },
  {
    label: "Restart gateway",
    command: "disabled:restart",
    risk: "critical",
    description: "Critical action. Must route through existing Hermes restart gate later.",
  },
];

export const recentLogs: LogLine[] = [
  { level: "info", time: "02:34", message: "Gate A validation passed: 203 tests." },
  { level: "info", time: "02:35", message: "Codex R2 review: OK." },
  { level: "warn", time: "02:37", message: "Runtime actions disabled until approve-gate milestone." },
];
