import { useCallback, useEffect, useMemo, useState } from "react";
import { hasMiniAppApi } from "./api";
import { navItems, statusCards, type NavKey } from "./mockData";
import {
  STORAGE_KEYS,
  actionValue,
  endpointKeys,
  formatRefreshTime,
  formatRuntime,
  freshnessState,
  gatewayMeta,
  gatewayValue,
  logLineKey,
  readStoredNavKey,
  writeStoredValue,
} from "./appModel";
import { configureTelegramBackButton, configureTelegramMainButton, getTelegramRuntime, prepareTelegramViewport } from "./telegram";
import { useMiniAppSnapshots } from "./useMiniAppSnapshots";
import { SectionIntro, SourceStrip } from "./components/chrome";
import { StatusSection } from "./components/StatusSection";
import { SessionsSection } from "./components/SessionsSection";
import { ApprovalsSection } from "./components/ApprovalsSection";
import { LogsSection } from "./components/LogsSection";
import { CommandPalette } from "./components/CommandPalette";

export function App() {
  const [activeTab, setActiveTab] = useState<NavKey>(() => readStoredNavKey());
  const [isPaletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const apiConfigured = hasMiniAppApi(telegram.isTelegram);
  const {
    snapshot,
    apiState,
    approvals,
    sessions,
    logs,
    serverCapabilities,
    endpointHealth,
    snapshotMeta,
    isRefreshing,
    lastSuccessAt,
    now,
    refreshSnapshots,
    selectedApprovalId,
    setSelectedApprovalId,
    selectedSessionId,
    setSelectedSessionId,
    selectedLogKey,
    setSelectedLogKey,
    logLevelFilter,
    setLogLevelFilter,
  } = useMiniAppSnapshots(telegram, apiConfigured);
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
    writeStoredValue(STORAGE_KEYS.activeTab, activeTab);
  }, [activeTab]);

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

      {activeTab === "status" ? <StatusSection cards={cards} safetyText={safetyText} approvalCount={approvalCount} capabilities={serverCapabilities} endpointHealth={endpointKeys.map((key) => endpointHealth[key])} /> : null}
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
