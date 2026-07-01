import { type CSSProperties, useEffect, useMemo, useState } from "react";
import { authenticateTelegram, fetchStatusSnapshot, hasMiniAppApi, type StatusSnapshot } from "./api";
import { navItems, quickActions, recentLogs, statusCards, type QuickAction } from "./mockData";
import { getTelegramRuntime, prepareTelegramViewport } from "./telegram";

const riskLabels: Record<QuickAction["risk"], string> = {
  safe: "безопасно",
  read_only: "только чтение",
  disabled: "отключено",
  critical: "критично",
};

function RiskBadge({ action }: { action: QuickAction }) {
  return <span className={`risk risk-${action.risk}`}>{riskLabels[action.risk]}</span>;
}

function RouteHint({ action }: { action: QuickAction }) {
  const label = action.risk === "critical" ? "заблокировано" : action.risk === "disabled" ? "скоро" : "только чтение";
  return <span className="route-hint">{label}</span>;
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

export function App() {
  const [snapshot, setSnapshot] = useState<StatusSnapshot | null>(null);
  const [apiState, setApiState] = useState<"mock" | "connecting" | "connected" | "offline">("mock");

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const apiConfigured = hasMiniAppApi(telegram.isTelegram);
  const publicSmoke = snapshot?.miniapp.public_exposure === true;

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
            Персональная диспетчерская для агентов, сессий и будущих одобрений. Сейчас режим только чтение.
          </p>
          <div className={`connection-banner state-${apiState}`}>{connectionLabel}</div>
        </div>
        <aside className="runtime-panel" aria-label="Пользователь и среда">
          <span>Контекст</span>
          <strong>{displayName}</strong>
          <small>{telegram.isTelegram ? "Telegram WebView" : formatRuntime(telegram.platform)}</small>
        </aside>
      </section>

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
        <span className="approval-lock">0</span>
      </section>

      <section className="command-card glass-card" aria-label="Быстрые действия">
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
              <RouteHint action={action} />
            </article>
          ))}
        </div>
      </section>

      <section className="intel-grid" aria-label="Будущие разделы">
        <article className="mini-panel glass-card">
          <div className="section-heading compact">
            <h2>Сессии</h2>
            <span className="risk risk-read_only">скоро</span>
          </div>
          <p className="muted">Здесь будет шкала событий Codex, Claude, OpenClaw и наблюдателей.</p>
        </article>
        <article className="mini-panel glass-card">
          <div className="section-heading compact">
            <h2>Журнал</h2>
            <span className="risk risk-disabled">редактировано</span>
          </div>
          <div className="log-list">
            {recentLogs.map((line) => (
              <p className={`log-line level-${line.level}`} key={`${line.time}-${line.message}`}>
                <span>{line.time}</span>
                {line.message}
              </p>
            ))}
          </div>
        </article>
      </section>

      <nav className="bottom-nav" aria-label="Навигация">
        {navItems.map((item) => (
          <button aria-current={item.active ? "page" : undefined} className={item.active ? "active" : ""} key={item.label} type="button" disabled>
            <span>{item.label}</span>
            {item.badge ? <em>{item.badge}</em> : null}
          </button>
        ))}
      </nav>
    </main>
  );
}
