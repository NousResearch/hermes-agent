import { type CSSProperties, useCallback, useEffect, useMemo, useState } from "react";
import { authenticateTelegram, fetchStatusSnapshot, hasMiniAppApi, type StatusSnapshot } from "./api";
import {
  approvalPreviews,
  navItems,
  quickActions,
  recentLogs,
  sessionPreviews,
  statusCards,
  type ApprovalPreview,
  type NavKey,
  type QuickAction,
} from "./mockData";
import { configureTelegramBackButton, configureTelegramMainButton, getTelegramRuntime, prepareTelegramViewport } from "./telegram";

const riskLabels: Record<QuickAction["risk"], string> = {
  safe: "безопасно",
  read_only: "только чтение",
  disabled: "отключено",
  critical: "критично",
};

const navTitles: Record<NavKey, string> = {
  status: "Операционный статус",
  sessions: "Сессии агентов",
  approvals: "Контур одобрений",
  logs: "Журнал событий",
};

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

function actionValue(snapshot: StatusSnapshot | null) {
  if (!snapshot) return "Блок";
  return snapshot.miniapp.actions_enabled ? "Вкл" : "Блок";
}

function SectionIntro({ activeTab, onOpenPalette }: { activeTab: NavKey; onOpenPalette: () => void }) {
  return (
    <section className="section-intro glass-card" aria-label="Текущий раздел">
      <div>
        <p className="mono-label">M4 / APP SHELL</p>
        <h2>{navTitles[activeTab]}</h2>
        <p>Навигация уже живая. Исполнение команд остаётся выключенным до серверного контура одобрений.</p>
      </div>
      <button className="palette-button tap" type="button" onClick={onOpenPalette}>
        Палитра
      </button>
    </section>
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

function SessionsSection() {
  return (
    <section className="stack-list" aria-label="Сессии агентов">
      {sessionPreviews.map((session) => (
        <article className={`list-card glass-card tone-${session.tone}`} key={session.id}>
          <div>
            <span className="mono-label">{session.time}</span>
            <h2>{session.agent}</h2>
            <p>{session.meta}</p>
          </div>
          <strong>{session.state}</strong>
        </article>
      ))}
    </section>
  );
}

function ApprovalsSection({ selectedId, onSelect }: { selectedId: string; onSelect: (approval: ApprovalPreview) => void }) {
  const selected = approvalPreviews.find((approval) => approval.id === selectedId) ?? approvalPreviews[0];

  return (
    <section className="approval-workspace" aria-label="Очередь одобрений">
      <div className="stack-list compact-stack">
        {approvalPreviews.map((approval) => (
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
        <div className="decision-strip" aria-label="Недоступные решения">
          <button type="button" disabled>Одобрить позже</button>
          <button type="button" disabled>Отклонить позже</button>
        </div>
      </article>
    </section>
  );
}

function LogsSection() {
  return (
    <section className="mini-panel glass-card full-panel" aria-label="Журнал событий">
      <div className="section-heading compact">
        <div>
          <p className="mono-label">РЕДАКТИРОВАННАЯ ШКАЛА</p>
          <h2>События без секретов</h2>
        </div>
        <span className="risk risk-read_only">только чтение</span>
      </div>
      <div className="log-list expanded">
        {recentLogs.map((line) => (
          <p className={`log-line level-${line.level}`} key={`${line.time}-${line.message}`}>
            <span>{line.time}</span>
            {line.message}
          </p>
        ))}
      </div>
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
  const [activeTab, setActiveTab] = useState<NavKey>("status");
  const [selectedApprovalId, setSelectedApprovalId] = useState(approvalPreviews[0]?.id ?? "");
  const [isPaletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const apiConfigured = hasMiniAppApi(telegram.isTelegram);
  const publicSmoke = snapshot?.miniapp.public_exposure === true;

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
    if (!telegram.isTelegram || !telegram.initData || !apiConfigured) {
      setApiState("mock");
      setSnapshot(null);
      return;
    }

    let cancelled = false;
    setApiState("connecting");
    authenticateTelegram(telegram.initData)
      .then(() => fetchStatusSnapshot())
      .then((status) => {
        if (!cancelled) {
          setSnapshot(status);
          setApiState("connected");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSnapshot(null);
          setApiState("offline");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [apiConfigured, telegram.initData, telegram.isTelegram]);

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
        value: "0",
        meta: "очередь пуста",
        tone: "ok",
      },
      {
        label: "Действия",
        value: actionValue(snapshot),
        meta: snapshot.miniapp.actions_enabled ? "нужен контур одобрений" : "безопасный режим",
        tone: "warn",
      },
    ] as const;
  }, [snapshot]);

  const connectionLabel =
    apiState === "connected"
      ? publicSmoke
        ? "HTTPS-проверка · только чтение"
        : "реальный статус · только чтение"
      : apiState === "connecting"
        ? "подключаю локальный сервис"
        : apiState === "offline"
          ? "локальный сервис недоступен · превью"
          : apiConfigured
            ? "API готов · жду Telegram initData"
            : "локальное превью";

  const safetyText = snapshot?.miniapp.actions_enabled
    ? "Сервер сообщил о включённых действиях. Перед любым запуском нужен контур одобрений."
    : "Ничего не будет выполнено без вашего явного одобрения.";
  const approvalCount = snapshot ? 0 : approvalPreviews.length;

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

      <SectionIntro activeTab={activeTab} onOpenPalette={() => setPaletteOpen(true)} />

      {activeTab === "status" ? <StatusSection cards={cards} safetyText={safetyText} approvalCount={approvalCount} /> : null}
      {activeTab === "sessions" ? <SessionsSection /> : null}
      {activeTab === "approvals" ? <ApprovalsSection selectedId={selectedApprovalId} onSelect={(approval) => setSelectedApprovalId(approval.id)} /> : null}
      {activeTab === "logs" ? <LogsSection /> : null}

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