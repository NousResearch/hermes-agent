import {
  type ApprovalItem,
  type LogPreviewItem,
  type SessionPreviewItem,
  type StatusSnapshot,
} from "./api";
import {
  type ApprovalPreview,
  type LogLine,
  type NavKey,
  type QuickAction,
  type SessionPreview,
} from "./mockData";

export const riskLabels: Record<QuickAction["risk"], string> = {
  safe: "безопасно",
  read_only: "только чтение",
  disabled: "отключено",
  critical: "критично",
};

export const POLL_INTERVAL_MS = 15_000;
export const STALE_AFTER_MS = 45_000;

export type FreshnessState = "mock" | "fresh" | "refreshing" | "stale" | "offline";
export type LogLevelFilter = LogLine["level"] | "all";
export type EndpointKey = "status" | "capabilities" | "approvals" | "sessions" | "logs";
export type EndpointState = "preview" | "checking" | "ok" | "degraded";
export type ApiState = "mock" | "connecting" | "connected" | "offline";

export type EndpointHealthItem = {
  key: EndpointKey;
  label: string;
  state: EndpointState;
  detail: string;
};

export const logLevelFilters: ReadonlyArray<{ key: LogLevelFilter; label: string }> = [
  { key: "all", label: "Все" },
  { key: "info", label: "Info" },
  { key: "warn", label: "Warn" },
  { key: "error", label: "Error" },
];

export const refreshLabels: Record<FreshnessState, string> = {
  mock: "локальное превью",
  fresh: "данные свежие",
  refreshing: "обновляю",
  stale: "данные устарели",
  offline: "сервис недоступен",
};

export const navTitles: Record<NavKey, string> = {
  status: "Операционный статус",
  sessions: "Сессии агентов",
  approvals: "Контур одобрений",
  logs: "Журнал событий",
};

export const endpointKeys: EndpointKey[] = ["status", "capabilities", "approvals", "sessions", "logs"];

export const endpointLabels: Record<EndpointKey, string> = {
  status: "Статус системы",
  capabilities: "Матрица возможностей",
  approvals: "Очередь одобрений",
  sessions: "Сессии",
  logs: "Логи",
};

export const endpointDetails: Record<EndpointState, string> = {
  preview: "локальное превью",
  checking: "проверяю read-only snapshot",
  ok: "read-only snapshot получен",
  degraded: "snapshot временно недоступен",
};

export const STORAGE_KEYS = {
  activeTab: "hermes-miniapp:active-tab",
  selectedApprovalId: "hermes-miniapp:selected-approval-id",
} as const;

export function isNavKey(value: string | null): value is NavKey {
  return value === "status" || value === "sessions" || value === "approvals" || value === "logs";
}

export function readStoredNavKey(): NavKey {
  try {
    const value = window.localStorage.getItem(STORAGE_KEYS.activeTab);
    return isNavKey(value) ? value : "status";
  } catch {
    return "status";
  }
}

export function readStoredApprovalId(fallback: string): string {
  try {
    return window.localStorage.getItem(STORAGE_KEYS.selectedApprovalId) || fallback;
  } catch {
    return fallback;
  }
}

export function writeStoredValue(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // localStorage is optional in Telegram WebView/private contexts.
  }
}

export function removeStoredValue(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // localStorage is optional in Telegram WebView/private contexts.
  }
}

export function createEndpointHealth(state: EndpointState): Record<EndpointKey, EndpointHealthItem> {
  return endpointKeys.reduce(
    (items, key) => ({
      ...items,
      [key]: {
        key,
        label: endpointLabels[key],
        state,
        detail: endpointDetails[state],
      },
    }),
    {} as Record<EndpointKey, EndpointHealthItem>,
  );
}

export function createEndpointStateUpdates(state: EndpointState): Record<EndpointKey, EndpointState> {
  return endpointKeys.reduce(
    (items, key) => ({
      ...items,
      [key]: state,
    }),
    {} as Record<EndpointKey, EndpointState>,
  );
}

export function updateEndpointHealth(current: Record<EndpointKey, EndpointHealthItem>, updates: Partial<Record<EndpointKey, EndpointState>>): Record<EndpointKey, EndpointHealthItem> {
  return endpointKeys.reduce(
    (items, key) => {
      const state = updates[key] ?? current[key].state;
      return {
        ...items,
        [key]: {
          key,
          label: endpointLabels[key],
          state,
          detail: endpointDetails[state],
        },
      };
    },
    {} as Record<EndpointKey, EndpointHealthItem>,
  );
}

export function formatRuntime(platform?: string) {
  if (!platform) return "среда неизвестна";
  if (platform === "unknown") return "среда неизвестна";
  return platform;
}

export function gatewayValue(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "Готов";
  return snapshot.gateway.running ? "Готов" : "Офлайн";
}

export function gatewayMeta(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "локальное превью";
  if (snapshot.gateway.busy) return "агенты работают";
  return snapshot.gateway.state || "стабильно";
}

export function formatRefreshTime(timestamp: number | null) {
  if (!timestamp) return "ещё не обновлялось";
  return new Intl.DateTimeFormat("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(timestamp);
}

export function freshnessState(apiState: ApiState, isRefreshing: boolean, lastSuccessAt: number | null, now: number): FreshnessState {
  if (apiState === "mock") return "mock";
  if (isRefreshing || apiState === "connecting") return "refreshing";
  if (apiState === "offline") return lastSuccessAt ? "stale" : "offline";
  if (lastSuccessAt && now - lastSuccessAt > STALE_AFTER_MS) return "stale";
  return "fresh";
}

export function actionValue(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "Блок";
  return snapshot.miniapp.actions_enabled ? "Вкл" : "Блок";
}

export function mapServerApproval(item: ApprovalItem): ApprovalPreview {
  return {
    id: item.id,
    title: item.title,
    source: item.source,
    risk: item.risk,
    summary: item.summary,
    requestedAt: item.requested_at,
    status: item.status === "waiting" ? "ожидает" : "заблокировано",
    checks: item.checks,
  };
}

export function mapServerSession(item: SessionPreviewItem): SessionPreview {
  const stateMap: Record<SessionPreviewItem["state"], SessionPreview["state"]> = {
    observing: "наблюдение",
    waiting: "ожидание",
    completed: "завершено",
  };
  return {
    id: item.id,
    agent: item.agent,
    state: stateMap[item.state],
    meta: item.meta,
    time: item.time,
    tone: item.tone,
  };
}

export function mapServerLog(item: LogPreviewItem): LogLine {
  return {
    level: item.level,
    message: item.message,
    time: item.time,
  };
}

export function logLineKey(line: LogLine): string {
  return `${line.time}-${line.level}-${line.message}`;
}
