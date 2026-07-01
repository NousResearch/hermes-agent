import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  authenticateTelegram,
  fetchApprovalsSnapshot,
  fetchLogsSnapshot,
  fetchSessionsSnapshot,
  fetchStatusSnapshot,
  hasMiniAppApi,
  type ApprovalItem,
  type LogPreviewItem,
  type SessionPreviewItem,
  type SnapshotMeta,
  type StatusSnapshot,
} from "./api";
import {
  approvalPreviews,
  navItems,
  quickActions,
  recentLogs,
  sessionPreviews,
  statusCards,
  type ApprovalPreview,
  type LogLine,
  type NavKey,
  type QuickAction,
  type SessionPreview,
} from "./mockData";
import { configureTelegramBackButton, configureTelegramMainButton, getTelegramRuntime, prepareTelegramViewport, triggerTelegramRefreshHaptic } from "./telegram";

const riskLabels: Record<QuickAction["risk"], string> = {
  safe: "безопасно",
  read_only: "только чтение",
  disabled: "отключено",
  critical: "критично",
};

const POLL_INTERVAL_MS = 15_000;
const STALE_AFTER_MS = 45_000;

type FreshnessState = "mock" | "fresh" | "refreshing" | "stale" | "offline";
type LogLevelFilter = LogLine["level"] | "all";

const logLevelFilters: ReadonlyArray<{ key: LogLevelFilter; label: string }> = [
  { key: "all", label: "Все" },
  { key: "info", label: "Info" },
  { key: "warn", label: "Warn" },
  { key: "error", label: "Error" },
];

const refreshLabels: Record<FreshnessState, string> = {
  mock: "локальное превью",
  fresh: "данные свежие",
  refreshing: "обновляю",
  stale: "данные устарели",
  offline: "сервис недоступен",
};

const navTitles: Record<NavKey, string> = {
  status: "Операционный статус",
  sessions: "Сессии агентов",
  approvals: "Контур одобрений",
  logs: "Журнал событий",
};

const STORAGE_KEYS = {
  activeTab: "hermes-miniapp:active-tab",
  selectedApprovalId: "hermes-miniapp:selected-approval-id",
} as const;

function isNavKey(value: string | null): value is NavKey {
  return value === "status" || value === "sessions" || value === "approvals" || value === "logs";
}

function readStoredNavKey(): NavKey {
  try {
    const value = window.localStorage.getItem(STORAGE_KEYS.activeTab);
    return isNavKey(value) ? value : "status";
  } catch {
    return "status";
  }
}

function readStoredApprovalId(fallback: string): string {
  try {
    return window.localStorage.getItem(STORAGE_KEYS.selectedApprovalId) || fallback;
  } catch {
    return fallback;
  }
}

function writeStoredValue(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // localStorage is optional in Telegram WebView/private contexts.
  }
}

function removeStoredValue(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // localStorage is optional in Telegram WebView/private contexts.
  }
}

function RiskBadge({ action }: { action: Pick<QuickAction, "risk"> }) {
  return <span className={`risk risk-${action.risk}`}>{riskLabels[action.risk]}</span>;
}

function formatRuntime(platform?: string) {
  if (!platform) return "среда неизвестна";
  if (platform === "unknown") return "среда неизвестна";
  return platform;
}

function gatewayValue(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "Готов";
  return snapshot.gateway.running ? "Готов" : "Офлайн";
}

function gatewayMeta(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "локальное превью";
  if (snapshot.gateway.busy) return "агенты работают";
  return snapshot.gateway.state || "стабильно";
}

function formatRefreshTime(timestamp: number | null) {
  if (!timestamp) return "ещё не обновлялось";
  return new Intl.DateTimeFormat("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(timestamp);
}

function freshnessState(apiState: "mock" | "connecting" | "connected" | "offline", isRefreshing: boolean, lastSuccessAt: number | null, now: number): FreshnessState {
  if (apiState === "mock") return "mock";
  if (isRefreshing || apiState === "connecting") return "refreshing";
  if (apiState === "offline") return lastSuccessAt ? "stale" : "offline";
  if (lastSuccessAt && now - lastSuccessAt > STALE_AFTER_MS) return "stale";
  return "fresh";
}

function actionValue(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "Блок";
  return snapshot.miniapp.actions_enabled ? "Вкл" : "Блок";
}

function mapServerApproval(item: ApprovalItem): ApprovalPreview {
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

function mapServerSession(item: SessionPreviewItem): SessionPreview {
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

function mapServerLog(item: LogPreviewItem): LogLine {
  return {
    level: item.level,
    message: item.message,
    time: item.time,
  };
}

function logLineKey(line: LogLine): string {
  return `${line.time}-${line.level}-${line.message}`;
}

function SectionIntro({
  activeTab,
  freshness,
  isRefreshEnabled,
  lastRefreshText,
  onOpenPalette,
  onRefresh,
}: {
  activeTab: NavKey;
  freshness: FreshnessState;
  isRefreshEnabled: boolean;
  lastRefreshText: string;
  onOpenPalette: () => void;
  onRefresh: () => void;
}) {
  return (
    <section className="section-intro glass-card" aria-label="Текущий раздел">
      <div>
        <p className="mono-label">M10 / LIVE READ-ONLY</p>
        <h2>{navTitles[activeTab]}</h2>
        <p>Данные обновляются только чтением. Ручное обновление не запускает команды и не меняет состояние Hermes.</p>
        <div className="freshness-row" data-state={freshness}>
          <span>{refreshLabels[freshness]}</span>
          <small>{lastRefreshText}</small>
        </div>
      </div>
      <div className="section-actions">
        <button className="refresh-button tap" type="button" disabled={!isRefreshEnabled || freshness === "refreshing"} onClick={onRefresh}>
          {freshness === "refreshing" ? "Обновляю" : "Обновить"}
        </button>
        <button className="palette-button tap" type="button" onClick={onOpenPalette}>
          Палитра
        </button>
      </div>
    </section>
  );
}

function SourceStrip({ meta }: { meta: SnapshotMeta | null }) {
  if (!meta) return null;

  return (
    <div className="source-strip" aria-label="Источник данных">
      <span>{meta.source_label}</span>
      <small>{meta.redaction}</small>
      <em>{meta.contains_live_actions ? "actions linked" : "no actions"}</em>
    </div>
  );
}

function StatusSection({
  cards,
  safetyText,
  approvalCount,
}: {
  cards: ReadonlyArray<{ label: string; value: string; meta: string; tone: string }>;
  safetyText: string;
  approvalCount: number;
}) {
  return (
    <>
      <section className="status-grid" aria-label="Ключевые состояния Hermes">
        {cards.map((card, index) => (
          <article className={`metric-card tone-${card.tone}`} key={card.label} style={{ "--delay": `${index * 55}ms` } as CSSProperties}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <small>{card.meta}</small>
          </article>
        ))}
      </section>

      <section className="approval-card glass-card" aria-label="Состояние одобрений">
        <div>
          <p className="mono-label">КОНТУР ОДОБРЕНИЙ</p>
          <h2>Опасные действия заблокированы</h2>
          <p>{safetyText}</p>
        </div>
        <span className="approval-lock">{approvalCount}</span>
      </section>

      <section className="command-card glass-card" aria-label="Быстрые маршруты">
        <div className="section-heading">
          <div>
            <p className="mono-label">КОМАНДНАЯ ПОВЕРХНОСТЬ</p>
            <h2>Безопасные маршруты</h2>
          </div>
          <span className="lock-pill">действия выключены</span>
        </div>

        <div className="quick-grid">
          {quickActions.map((action, index) => (
            <article className="quick-tile tap" key={action.id} style={{ "--delay": `${index * 45}ms` } as CSSProperties}>
              <div className="action-title">
                <strong>{action.label}</strong>
                <RiskBadge action={action} />
              </div>
              <p>{action.description}</p>
              <span className="route-hint">{action.hint}</span>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <section className="empty-state glass-card" aria-label={title}>
      <p className="mono-label">ПУСТОЙ SERVER SNAPSHOT</p>
      <h2>{title}</h2>
      <p>{text}</p>
    </section>
  );
}

function SessionsSection({ sessions, selectedId, onSelect }: { sessions: SessionPreview[]; selectedId: string; onSelect: (session: SessionPreview) => void }) {
  if (sessions.length === 0) {
    return <EmptyState title="Сессий нет" text="Сервер ответил пустым списком. Локальные mock-данные не подмешиваются после успешного обновления." />;
  }

  const selected = sessions.find((session) => session.id === selectedId) ?? sessions[0];

  return (
    <section className="drilldown-workspace" aria-label="Сессии агентов">
      <div className="stack-list compact-stack">
        {sessions.map((session) => (
          <button className={`list-card glass-card tap tone-${session.tone}`} data-selected={session.id === selected.id} key={session.id} type="button" onClick={() => onSelect(session)}>
            <div>
              <span className="mono-label">{session.time}</span>
              <h2>{session.agent}</h2>
              <p>{session.meta}</p>
            </div>
            <strong>{session.state}</strong>
          </button>
        ))}
      </div>

      <article className="drilldown-detail glass-card" aria-label="Деталь выбранной сессии">
        <div className="section-heading compact">
          <div>
            <p className="mono-label">M10 / SESSION DETAIL</p>
            <h2>{selected.agent}</h2>
          </div>
          <span className="risk risk-read_only">только чтение</span>
        </div>
        <p>{selected.meta}</p>
        <div className="detail-meta">
          <span>{selected.state}</span>
          <span>{selected.time}</span>
        </div>
        <div className="readonly-note">
          <strong>Наблюдение без команд</strong>
          <small>Этот экран не открывает терминал, не меняет процессы и не отправляет команды агентам.</small>
        </div>
      </article>
    </section>
  );
}

function DecisionReadiness({ approval }: { approval: ApprovalPreview }) {
  const guardrails = [
    {
      label: "Действия сервера",
      value: "не подключены",
      ready: false,
      detail: "В Mini App нет endpoint для approve/reject/restart.",
    },
    {
      label: "Контекст владельца",
      value: approval.risk === "critical" ? "требуется" : "ожидает",
      ready: false,
      detail: "Будущее решение должно пройти ручное подтверждение Андрея.",
    },
    {
      label: "Проверки запроса",
      value: `${approval.checks.length} видно`,
      ready: approval.checks.length > 0,
      detail: "Список отображается только как read-only preflight.",
    },
  ] as const;

  return (
    <section className="decision-readiness" aria-label="Готовность будущего решения">
      <div className="decision-readiness-head">
        <div>
          <p className="mono-label">M9 / PREFLIGHT</p>
          <h3>Решение ещё не подключено</h3>
        </div>
        <span className="lock-pill">read-only</span>
      </div>
      <p>Этот блок заранее показывает, что должно быть проверено перед будущим approve/reject. Сейчас он ничего не отправляет и не меняет.</p>
      <div className="readiness-grid">
        {guardrails.map((item) => (
          <article className="readiness-item" data-ready={item.ready} key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <small>{item.detail}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function ApprovalsSection({ approvals, selectedId, onSelect }: { approvals: ApprovalPreview[]; selectedId: string; onSelect: (approval: ApprovalPreview) => void }) {
  if (approvals.length === 0) {
    return <EmptyState title="Очередь одобрений пуста" text="Сервер вернул пустую очередь. Это считается валидным live-состоянием, а не поводом показывать mock-запросы." />;
  }

  const selected = approvals.find((approval) => approval.id === selectedId) ?? approvals[0];

  return (
    <section className="approval-workspace" aria-label="Очередь одобрений">
      <div className="stack-list compact-stack">
        {approvals.map((approval) => (
          <button className="approval-row glass-card tap" data-selected={approval.id === selected.id} key={approval.id} type="button" onClick={() => onSelect(approval)}>
            <span>
              <strong>{approval.title}</strong>
              <small>{approval.source}</small>
            </span>
            <RiskBadge action={approval} />
          </button>
        ))}
      </div>

      <article className="approval-detail glass-card">
        <div className="section-heading compact">
          <div>
            <p className="mono-label">ДЕТАЛЬ ЗАПРОСА</p>
            <h2>{selected.title}</h2>
          </div>
          <RiskBadge action={selected} />
        </div>
        <p>{selected.summary}</p>
        <div className="detail-meta">
          <span>{selected.status}</span>
          <span>{selected.requestedAt}</span>
        </div>
        <ul className="check-list">
          {selected.checks.map((check) => (
            <li key={check}>{check}</li>
          ))}
        </ul>
        <DecisionReadiness approval={selected} />
        <div className="decision-strip" aria-label="Недоступные решения">
          <button type="button" disabled>Одобрить позже</button>
          <button type="button" disabled>Отклонить позже</button>
        </div>
      </article>
    </section>
  );
}

function LogsSection({
  logs,
  selectedKey,
  levelFilter,
  onFilterChange,
  onSelect,
}: {
  logs: LogLine[];
  selectedKey: string;
  levelFilter: LogLevelFilter;
  onFilterChange: (level: LogLevelFilter) => void;
  onSelect: (line: LogLine) => void;
}) {
  if (logs.length === 0) {
    return <EmptyState title="Журнал пуст" text="Серверный журнал сейчас пуст. После live-ответа локальные демонстрационные события скрыты." />;
  }

  const filteredLogs = levelFilter === "all" ? logs : logs.filter((line) => line.level === levelFilter);
  const selected = filteredLogs.find((line) => logLineKey(line) === selectedKey) ?? filteredLogs[0] ?? null;

  return (
    <section className="logs-workspace" aria-label="Журнал событий">
      <div className="mini-panel glass-card full-panel">
        <div className="section-heading compact">
          <div>
            <p className="mono-label">РЕДАКТИРОВАННАЯ ШКАЛА</p>
            <h2>События без секретов</h2>
          </div>
          <span className="risk risk-read_only">только чтение</span>
        </div>
        <div className="filter-chips" aria-label="Фильтр уровня логов">
          {logLevelFilters.map((filter) => (
            <button aria-pressed={levelFilter === filter.key} className="filter-chip tap" key={filter.key} type="button" onClick={() => onFilterChange(filter.key)}>
              {filter.label}
            </button>
          ))}
        </div>
        {filteredLogs.length === 0 ? (
          <div className="inline-empty">
            <strong>Нет событий этого уровня</strong>
            <small>Фильтр работает только по уже редактированным строкам, без запроса новых данных.</small>
          </div>
        ) : (
          <div className="log-list expanded">
            {filteredLogs.map((line) => {
              const key = logLineKey(line);
              return (
                <button className={`log-line tap level-${line.level}`} data-selected={key === selectedKey} key={key} type="button" onClick={() => onSelect(line)}>
                  <span>{line.time}</span>
                  {line.message}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {selected ? (
        <article className="drilldown-detail glass-card" aria-label="Деталь выбранного события">
          <div className="section-heading compact">
            <div>
              <p className="mono-label">M10 / LOG DETAIL</p>
              <h2>{selected.level.toUpperCase()}</h2>
            </div>
            <span className={`risk risk-${selected.level === "info" ? "read_only" : selected.level === "warn" ? "disabled" : "critical"}`}>{selected.level}</span>
          </div>
          <p>{selected.message}</p>
          <div className="detail-meta">
            <span>{selected.time}</span>
            <span>redacted</span>
          </div>
          <div className="readonly-note">
            <strong>Секреты не показываются</strong>
            <small>Mini App отображает только allowlisted preview строки, без raw логов, токенов, env и путей.</small>
          </div>
        </article>
      ) : null}
    </section>
  );
}

function CommandPalette({ isOpen, onClose, onNavigate }: { isOpen: boolean; onClose: () => void; onNavigate: (tab: NavKey) => void }) {
  if (!isOpen) return null;

  const routeMap: Partial<Record<QuickAction["id"], NavKey>> = {
    status: "status",
    sessions: "sessions",
    approvals: "approvals",
    logs: "logs",
  };

  return (
    <div className="sheet-backdrop" role="presentation" onClick={onClose}>
      <section className="command-sheet glass-card" aria-label="Палитра команд" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
        <div className="sheet-handle" />
        <div className="section-heading compact">
          <div>
            <p className="mono-label">ПАЛИТРА</p>
            <h2>Быстрый переход</h2>
          </div>
          <button className="ghost-button" type="button" onClick={onClose}>Закрыть</button>
        </div>
        <div className="palette-list">
          {quickActions.map((action) => {
            const target = routeMap[action.id];
            return (
              <button
                className="palette-row tap"
                disabled={!target || action.risk === "critical"}
                key={action.id}
                type="button"
                onClick={() => {
                  if (!target) return;
                  onNavigate(target);
                  onClose();
                }}
              >
                <span>
                  <strong>{action.label}</strong>
                  <small>{action.description}</small>
                </span>
                <em>{action.hint}</em>
              </button>
            );
          })}
        </div>
      </section>
    </div>
  );
}

export function App() {
  const [snapshot, setSnapshot] = useState<StatusSnapshot | null>(null);
  const [apiState, setApiState] = useState<"mock" | "connecting" | "connected" | "offline">("mock");
  const [activeTab, setActiveTab] = useState<NavKey>(() => readStoredNavKey());
  const [selectedApprovalId, setSelectedApprovalId] = useState(() => readStoredApprovalId(approvalPreviews[0]?.id ?? ""));
  const [selectedSessionId, setSelectedSessionId] = useState(() => sessionPreviews[0]?.id ?? "");
  const [selectedLogKey, setSelectedLogKey] = useState(() => (recentLogs[0] ? logLineKey(recentLogs[0]) : ""));
  const [logLevelFilter, setLogLevelFilter] = useState<LogLevelFilter>("all");
  const [serverApprovals, setServerApprovals] = useState<ApprovalPreview[]>([]);
  const [serverSessions, setServerSessions] = useState<SessionPreview[]>([]);
  const [serverLogs, setServerLogs] = useState<LogLine[]>([]);
  const [snapshotMeta, setSnapshotMeta] = useState<Record<"approvals" | "sessions" | "logs", SnapshotMeta | null>>({ approvals: null, sessions: null, logs: null });
  const [isPaletteOpen, setPaletteOpen] = useState(false);
  const [isRefreshing, setRefreshing] = useState(false);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const refreshRequestId = useRef(0);
  const activeRefreshes = useRef(0);

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const apiConfigured = hasMiniAppApi(telegram.isTelegram);
  const publicSmoke = snapshot?.miniapp.public_exposure === true;
  const hasServerSnapshot = lastSuccessAt !== null;
  const approvals = hasServerSnapshot ? serverApprovals : approvalPreviews;
  const sessions = hasServerSnapshot ? serverSessions : sessionPreviews;
  const logs = hasServerSnapshot ? serverLogs : recentLogs;

  const closeTransientUi = useCallback(() => {
    if (isPaletteOpen) {
      setPaletteOpen(false);
      return;
    }
    if (activeTab !== "status") setActiveTab("status");
  }, [activeTab, isPaletteOpen]);

  useEffect(() => configureTelegramBackButton(isPaletteOpen || activeTab !== "status", closeTransientUi), [activeTab, closeTransientUi, isPaletteOpen]);
  useEffect(() => configureTelegramMainButton("Только чтение", () => undefined), []);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.activeTab, activeTab);
  }, [activeTab]);

  useEffect(() => {
    if (approvals.length === 0) {
      if (selectedApprovalId) setSelectedApprovalId("");
      removeStoredValue(STORAGE_KEYS.selectedApprovalId);
      return;
    }

    const selectedExists = approvals.some((approval) => approval.id === selectedApprovalId);
    if (!selectedExists) {
      setSelectedApprovalId(approvals[0].id);
      return;
    }

    writeStoredValue(STORAGE_KEYS.selectedApprovalId, selectedApprovalId);
  }, [approvals, selectedApprovalId]);

  useEffect(() => {
    if (sessions.length === 0) {
      if (selectedSessionId) setSelectedSessionId("");
      return;
    }

    const selectedExists = sessions.some((session) => session.id === selectedSessionId);
    if (!selectedExists) setSelectedSessionId(sessions[0].id);
  }, [sessions, selectedSessionId]);

  useEffect(() => {
    if (logs.length === 0) {
      if (selectedLogKey) setSelectedLogKey("");
      return;
    }

    const filteredLogs = logLevelFilter === "all" ? logs : logs.filter((line) => line.level === logLevelFilter);
    const selectedExists = filteredLogs.some((line) => logLineKey(line) === selectedLogKey);
    if (!selectedExists) setSelectedLogKey(filteredLogs[0] ? logLineKey(filteredLogs[0]) : "");
  }, [logs, logLevelFilter, selectedLogKey]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5_000);
    return () => window.clearInterval(timer);
  }, []);

  const refreshSnapshots = useCallback(
    async ({ manual = false }: { manual?: boolean } = {}) => {
      if (!telegram.isTelegram || !telegram.initData || !apiConfigured) return;

      const requestId = refreshRequestId.current + 1;
      refreshRequestId.current = requestId;
      activeRefreshes.current += 1;
      if (manual) triggerTelegramRefreshHaptic("start");
      setRefreshing(true);
      setApiState((current) => (current === "connected" ? current : "connecting"));

      try {
        const [status, approvals, sessions, logs] = await Promise.all([fetchStatusSnapshot(), fetchApprovalsSnapshot(), fetchSessionsSnapshot(), fetchLogsSnapshot()]);
        if (requestId !== refreshRequestId.current) return;

        setSnapshot(status);
        const mappedApprovals = approvals.items.map(mapServerApproval);
        const mappedSessions = sessions.items.map(mapServerSession);
        const mappedLogs = logs.items.map(mapServerLog);
        setServerApprovals(mappedApprovals);
        setServerSessions(mappedSessions);
        setServerLogs(mappedLogs);
        setSnapshotMeta({ approvals: approvals.meta ?? null, sessions: sessions.meta ?? null, logs: logs.meta ?? null });
        setSelectedApprovalId((current) => {
          if (mappedApprovals.some((approval) => approval.id === current)) return current;
          return mappedApprovals[0]?.id ?? "";
        });
        setSelectedSessionId((current) => {
          if (mappedSessions.some((session) => session.id === current)) return current;
          return mappedSessions[0]?.id ?? "";
        });
        setSelectedLogKey((current) => {
          if (mappedLogs.some((line) => logLineKey(line) === current)) return current;
          return mappedLogs[0] ? logLineKey(mappedLogs[0]) : "";
        });
        const refreshedAt = Date.now();
        setLastSuccessAt(refreshedAt);
        setNow(refreshedAt);
        setApiState("connected");
        if (manual) triggerTelegramRefreshHaptic("success");
      } catch {
        if (requestId === refreshRequestId.current) {
          setApiState("offline");
          if (manual) triggerTelegramRefreshHaptic("warning");
        }
      } finally {
        activeRefreshes.current = Math.max(0, activeRefreshes.current - 1);
        if (activeRefreshes.current === 0) setRefreshing(false);
      }
    },
    [apiConfigured, telegram.initData, telegram.isTelegram],
  );

  useEffect(() => {
    if (!telegram.isTelegram || !telegram.initData || !apiConfigured) {
      setApiState("mock");
      setSnapshot(null);
      setServerApprovals([]);
      setServerSessions([]);
      setServerLogs([]);
      setSnapshotMeta({ approvals: null, sessions: null, logs: null });
      setLastSuccessAt(null);
      setRefreshing(false);
      return;
    }

    let cancelled = false;
    setApiState("connecting");
    setRefreshing(true);
    void authenticateTelegram(telegram.initData)
      .then(() => {
        if (!cancelled) return refreshSnapshots();
        return undefined;
      })
      .catch(() => {
        if (!cancelled) {
          setApiState("offline");
          setRefreshing(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [apiConfigured, refreshSnapshots, telegram.initData, telegram.isTelegram]);

  useEffect(() => {
    if (!telegram.isTelegram || !telegram.initData || !apiConfigured) return undefined;
    const timer = window.setInterval(() => {
      void refreshSnapshots();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [apiConfigured, refreshSnapshots, telegram.initData, telegram.isTelegram]);

  const displayName = telegram.user?.username
    ? `@${telegram.user.username}`
    : telegram.user?.first_name ?? "локальное превью";

  const cards = useMemo(() => {
    if (!snapshot) return statusCards;
    return [
      {
        label: "Шлюз",
        value: gatewayValue(snapshot),
        meta: gatewayMeta(snapshot),
        tone: snapshot.gateway.running ? "ok" : "danger",
      },
      {
        label: "Сессии",
        value: String(snapshot.gateway.active_agents),
        meta: snapshot.gateway.busy ? "активные агенты" : "нет активной нагрузки",
        tone: snapshot.gateway.busy ? "warn" : "muted",
      },
      {
        label: "Одобрения",
        value: String(approvals.length),
        meta: approvals.length > 0 ? "очередь на чтение" : "очередь пуста",
        tone: approvals.length > 0 ? "warn" : "ok",
      },
      {
        label: "Действия",
        value: actionValue(snapshot),
        meta: snapshot.miniapp.actions_enabled ? "нужен контур одобрений" : "безопасный режим",
        tone: "warn",
      },
    ] as const;
  }, [approvals.length, snapshot]);

  const connectionLabel =
    apiState === "connected"
      ? publicSmoke
        ? "HTTPS-проверка · только чтение"
        : "реальный статус · только чтение"
      : apiState === "connecting"
        ? "подключаю локальный сервис"
        : apiState === "offline"
          ? lastSuccessAt
            ? "локальный сервис недоступен · сохранён последний статус"
            : "локальный сервис недоступен · превью"
          : apiConfigured
            ? "API готов · жду Telegram initData"
            : "локальное превью";

  const safetyText = snapshot?.miniapp.actions_enabled
    ? "Сервер сообщил о включённых действиях. Перед любым запуском нужен контур одобрений."
    : "Ничего не будет выполнено без вашего явного одобрения.";
  const approvalCount = approvals.length;
  const currentFreshness = freshnessState(apiState, isRefreshing, lastSuccessAt, now);
  const lastRefreshText = lastSuccessAt ? `последнее обновление ${formatRefreshTime(lastSuccessAt)}` : formatRefreshTime(lastSuccessAt);
  const refreshEnabled = telegram.isTelegram && Boolean(telegram.initData) && apiConfigured;
  const currentSourceMeta = activeTab === "approvals" || activeTab === "sessions" || activeTab === "logs" ? snapshotMeta[activeTab] : null;

  return (
    <main className="screen" data-mode={telegram.colorScheme}>
      <div className="orb orb-cyan" />
      <div className="orb orb-gold" />
      <div className="grid-noise" />

      <header className="topbar" aria-label="Контекст Mini App">
        <div>
          <span className="mono-label">HERMES / MINI APP</span>
          <strong>Пульт управления</strong>
        </div>
        <span className="status-pill"><span />Защищённый режим</span>
      </header>

      <section className="hero-card glass-card scan-line">
        <div className="hero-copy-block">
          <p className="eyebrow">Светлая кибернетическая ОС</p>
          <h1>Hermes готов</h1>
          <p className="hero-copy">
            Персональная диспетчерская для агентов, сессий, логов и будущих ручных одобрений. Сейчас режим только чтение.
          </p>
          <div className={`connection-banner state-${apiState}`}>{connectionLabel}</div>
        </div>
        <aside className="runtime-panel" aria-label="Пользователь и среда">
          <span>Контекст</span>
          <strong>{displayName}</strong>
          <small>{telegram.isTelegram ? "Telegram WebView" : formatRuntime(telegram.platform)}</small>
        </aside>
      </section>

      <SectionIntro
        activeTab={activeTab}
        freshness={currentFreshness}
        isRefreshEnabled={refreshEnabled}
        lastRefreshText={lastRefreshText}
        onOpenPalette={() => setPaletteOpen(true)}
        onRefresh={() => void refreshSnapshots({ manual: true })}
      />
      <SourceStrip meta={currentSourceMeta} />

      {activeTab === "status" ? <StatusSection cards={cards} safetyText={safetyText} approvalCount={approvalCount} /> : null}
      {activeTab === "sessions" ? <SessionsSection sessions={sessions} selectedId={selectedSessionId} onSelect={(session) => setSelectedSessionId(session.id)} /> : null}
      {activeTab === "approvals" ? <ApprovalsSection approvals={approvals} selectedId={selectedApprovalId} onSelect={(approval) => setSelectedApprovalId(approval.id)} /> : null}
      {activeTab === "logs" ? (
        <LogsSection
          logs={logs}
          selectedKey={selectedLogKey}
          levelFilter={logLevelFilter}
          onFilterChange={setLogLevelFilter}
          onSelect={(line) => setSelectedLogKey(logLineKey(line))}
        />
      ) : null}

      <nav className="bottom-nav" aria-label="Навигация">
        {navItems.map((item) => (
          <button aria-current={activeTab === item.key ? "page" : undefined} className={activeTab === item.key ? "active" : ""} key={item.key} type="button" onClick={() => setActiveTab(item.key)}>
            <span>{item.label}</span>
            {item.badge ? <em>{item.badge}</em> : null}
          </button>
        ))}
      </nav>

      <CommandPalette isOpen={isPaletteOpen} onClose={() => setPaletteOpen(false)} onNavigate={setActiveTab} />
    </main>
  );
}