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
};

export type LogLine = {
  level: "info" | "warn" | "error";
  message: string;
  time: string;
};

export type NavItem = {
  label: string;
  badge?: string;
  active?: boolean;
};

export const statusCards: StatusCard[] = [
  { label: "Шлюз", value: "Готов", meta: "локальное превью", tone: "ok" },
  { label: "Сессии", value: "0", meta: "активных запусков", tone: "muted" },
  { label: "Одобрения", value: "0", meta: "нет запросов", tone: "ok" },
  { label: "Действия", value: "Блок", meta: "контур одобрений позже", tone: "warn" },
];

export const quickActions: QuickAction[] = [
  {
    id: "status",
    label: "Статус системы",
    risk: "read_only",
    description: "Обновить безопасный снимок состояния Hermes.",
  },
  {
    id: "sessions",
    label: "Сессии агентов",
    risk: "read_only",
    description: "Открыть будущий список Codex, Claude, OpenClaw и наблюдателей.",
  },
  {
    id: "logs",
    label: "Журнал событий",
    risk: "read_only",
    description: "Посмотреть шкалу событий без секретов и токенов.",
  },
  {
    id: "restart",
    label: "Перезапуск Hermes",
    risk: "critical",
    description: "Заблокировано до серверного контура одобрений.",
  },
];

export const recentLogs: LogLine[] = [
  { level: "info", time: "M2", message: "Status API работает только на чтение." },
  { level: "info", time: "M3", message: "Telegram initData проверяется на сервере." },
  { level: "warn", time: "M3", message: "Опасные действия заблокированы до контура одобрений." },
];

export const navItems: NavItem[] = [
  { label: "Статус", active: true },
  { label: "Сессии" },
  { label: "Одобрения", badge: "0" },
  { label: "Логи" },
];
