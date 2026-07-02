export type RiskLevel = "safe" | "read_only" | "disabled" | "critical";

export type StatusCard = {
  label: string;
  value: string;
  meta: string;
  tone: "ok" | "warn" | "muted" | "danger";
};

export type QuickAction = {
  id: string;
  label: string;
  risk: RiskLevel;
  description: string;
  hint: string;
};

export type LogLine = {
  level: "info" | "warn" | "error";
  message: string;
  time: string;
};

export type NavKey = "status" | "sessions" | "approvals" | "logs";

export type NavItem = {
  key: NavKey;
  label: string;
  badge?: string;
};

export type SessionPreview = {
  id: string;
  agent: string;
  state: "наблюдение" | "ожидание" | "завершено";
  meta: string;
  time: string;
  tone: "ok" | "warn" | "muted";
};

export type ApprovalDecisionValue = "approve_once" | "reject_once";

export type ApprovalPreview = {
  id: string;
  title: string;
  source: string;
  risk: RiskLevel;
  summary: string;
  requestedAt: string;
  status: "ожидает" | "заблокировано";
  checks: string[];
  allowedDecisions?: ApprovalDecisionValue[];
};

export const statusCards: StatusCard[] = [
  { label: "Шлюз", value: "Готов", meta: "локальное превью", tone: "ok" },
  { label: "Сессии", value: "0", meta: "активных запусков", tone: "muted" },
  { label: "Одобрения", value: "2", meta: "макет очереди", tone: "warn" },
  { label: "Действия", value: "Блок", meta: "контур одобрений позже", tone: "warn" },
];

export const quickActions: QuickAction[] = [
  {
    id: "status",
    label: "Статус системы",
    risk: "read_only",
    description: "Обновить безопасный снимок состояния Hermes.",
    hint: "только чтение",
  },
  {
    id: "sessions",
    label: "Сессии агентов",
    risk: "read_only",
    description: "Открыть будущий список Codex, Claude, OpenClaw и наблюдателей.",
    hint: "экран готов",
  },
  {
    id: "approvals",
    label: "Очередь одобрений",
    risk: "read_only",
    description: "Открыть безопасный макет будущего ручного решения.",
    hint: "макет",
  },
  {
    id: "logs",
    label: "Журнал событий",
    risk: "read_only",
    description: "Посмотреть шкалу событий без секретов и токенов.",
    hint: "без секретов",
  },
];

export const recentLogs: LogLine[] = [
  { level: "info", time: "M2", message: "Status API работает только на чтение." },
  { level: "info", time: "M3", message: "Telegram initData проверяется на сервере." },
  { level: "warn", time: "M4", message: "Интерфейс одобрений показан как безопасный макет." },
  { level: "warn", time: "M4", message: "Опасные действия всё ещё заблокированы сервером." },
];

export const navItems: NavItem[] = [
  { key: "status", label: "Статус" },
  { key: "sessions", label: "Сессии" },
  { key: "approvals", label: "Одобрения", badge: "2" },
  { key: "logs", label: "Логи" },
];

export const sessionPreviews: SessionPreview[] = [
  {
    id: "codex-review",
    agent: "Codex review",
    state: "завершено",
    meta: "проверка интерфейса и safety-copy",
    time: "12 мин назад",
    tone: "ok",
  },
  {
    id: "miniapp-watch",
    agent: "Mini App sidecar",
    state: "наблюдение",
    meta: "готов к локальному статусу, публичный режим выключен",
    time: "сейчас",
    tone: "warn",
  },
  {
    id: "openclaw",
    agent: "OpenClaw workspace",
    state: "ожидание",
    meta: "без активных команд из Mini App",
    time: "фон",
    tone: "muted",
  },
];

export const approvalPreviews: ApprovalPreview[] = [
  {
    id: "system-mode-change-preview",
    title: "Изменить системный режим",
    source: "будущий системный запрос",
    risk: "critical",
    summary: "Макет высокорискового действия перед ручным решением владельца.",
    requestedAt: "макет",
    status: "заблокировано",
    checks: ["нужен владелец", "нужна причина", "нужен rollback-план"],
  },
  {
    id: "read-logs",
    title: "Открыть журнал событий",
    source: "безопасный маршрут",
    risk: "read_only",
    summary: "Действие только читает редактированную шкалу событий без секретов.",
    requestedAt: "сейчас",
    status: "ожидает",
    checks: ["секреты скрыты", "команды не выполняются", "доступ только владельцу"],
  },
];