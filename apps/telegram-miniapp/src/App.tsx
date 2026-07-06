import { useCallback, useEffect, useMemo, useState } from "react";
import { hasMiniAppApi } from "./api";
import { navItems, statusCards, type NavKey } from "./mockData";
import {
  DECK_VERSION,
  STORAGE_KEYS,
  actionFooterLabel,
  actionValue,
  endpointKeys,
  formatRefreshTime,
  formatRuntime,
  freshnessState,
  gatewayMeta,
  gatewayValue,
  logLineKey,
  readStoredNavKey,
  readStoredTheme,
  writeStoredValue,
  type Theme,
} from "./appModel";
import { configureTelegramBackButton, getTelegramRuntime, prepareTelegramViewport } from "./telegram";
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
  const [theme, setTheme] = useState<Theme>(() => readStoredTheme());

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#20160f" : "#efe9dd");
    writeStoredValue(STORAGE_KEYS.theme, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
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
    endpointLastOk,
    snapshotMeta,
    isRefreshing,
    lastSuccessAt,
    now,
    refreshSnapshots,
    actionsEnabled,
    canSubmitDecision,
    decisionBlockReason,
    decisionError,
    clearDecisionError,
    isActionOwner,
    submitApprovalDecision,
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
  const isConnected = apiState === "connected";
  // The hero headline tracks the real connection/gateway state instead of a
  // static "Hermes готов", so the biggest signal on screen never lies.
  const heroTitle =
    apiState === "connected"
      ? snapshot?.gateway.running
        ? "Hermes на связи"
        : "Hermes офлайн"
      : apiState === "connecting"
        ? "Подключаюсь…"
        : apiState === "offline"
          ? lastSuccessAt
            ? "Связь потеряна"
            : "Hermes недоступен"
          : "Локальное превью";
  const currentFreshness = freshnessState(apiState, isRefreshing, lastSuccessAt, now);
  const lastRefreshText = lastSuccessAt ? `последнее обновление ${formatRefreshTime(lastSuccessAt)}` : formatRefreshTime(lastSuccessAt);
  const refreshEnabled = telegram.isTelegram && Boolean(telegram.initData) && apiConfigured;
  const currentSourceMeta = activeTab === "approvals" || activeTab === "sessions" || activeTab === "logs" ? snapshotMeta[activeTab] : null;
  // A live section renders only once it has committed data at least once; a
  // never-loaded degraded endpoint shows the degraded notice alone instead of
  // an empty-state that would masquerade an outage as a valid empty response.
  const sectionAvailable = activeTab === "status" || apiState === "mock" || endpointLastOk[activeTab] !== null;

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
        <div className="topbar-actions">
          <button
            className="theme-toggle tap"
            type="button"
            onClick={toggleTheme}
            aria-pressed={theme === "dark"}
            aria-label="Тёмная тема"
            title={theme === "dark" ? "Переключить на светлую" : "Переключить на тёмную"}
          >
            <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
          </button>
          <span className="status-pill"><span />Защищено</span>
        </div>
      </header>

      <section className="hero-card glass-card scan-line">
        <div className="hero-copy-block">
          <p className="eyebrow">Тихая роскошь · кибернетика</p>
          <h1>{heroTitle}</h1>
          <p className="hero-copy">
            Персональная диспетчерская для агентов, сессий, логов и будущих ручных одобрений. Сейчас режим только чтение.
          </p>
          <div className={`connection-banner state-${apiState}`} role="status" aria-live="polite">{connectionLabel}</div>
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

      {activeTab !== "status" && endpointHealth[activeTab].state === "degraded" ? (
        <div className="degraded-notice" role="status" aria-live="polite">
          <strong>Раздел не обновился в последнем опросе</strong>
          <small>
            {endpointLastOk[activeTab]
              ? `Показаны данные за ${formatRefreshTime(endpointLastOk[activeTab])}.`
              : "Этот раздел ещё ни разу не загрузился успешно."}
          </small>
        </div>
      ) : null}

      {activeTab === "status" ? <StatusSection cards={cards} safetyText={safetyText} approvalCount={approvalCount} capabilities={serverCapabilities} endpointHealth={endpointKeys.map((key) => endpointHealth[key])} /> : null}
      {activeTab === "sessions" && sectionAvailable ? <SessionsSection sessions={sessions} selectedId={selectedSessionId} onSelect={(session) => setSelectedSessionId(session.id)} /> : null}
      {activeTab === "approvals" && sectionAvailable ? (
        <ApprovalsSection
          approvals={approvals}
          selectedId={selectedApprovalId}
          onSelect={(approval) => setSelectedApprovalId(approval.id)}
          actionsEnabled={actionsEnabled}
          canSubmitDecision={canSubmitDecision}
          decisionBlockReason={decisionBlockReason}
          decisionError={decisionError}
          onClearDecisionError={clearDecisionError}
          isOwner={isActionOwner}
          isConnected={isConnected}
          onDecision={submitApprovalDecision}
        />
      ) : null}
      {activeTab === "logs" && sectionAvailable ? (
        <LogsSection
          logs={logs}
          selectedKey={selectedLogKey}
          levelFilter={logLevelFilter}
          onFilterChange={setLogLevelFilter}
          onSelect={(line) => setSelectedLogKey(logLineKey(line))}
        />
      ) : null}

      <footer className="deck-footer" aria-label="Версия и режим">
        <span>{DECK_VERSION}</span>
        <span>{actionFooterLabel(snapshot)}</span>
      </footer>

      <nav className="bottom-nav" aria-label="Навигация">
        {navItems.map((item) => {
          // The approvals badge is derived live from the real queue length, so
          // it shows the true number of pending items and disappears at zero.
          // In live mode it is suppressed unless the approvals endpoint is fresh
          // (state "ok"), so a degraded poll never shows a stale count as current.
          const approvalsTrustworthy = apiState === "mock" || endpointHealth.approvals.state === "ok";
          const badge = item.key === "approvals" && approvalCount > 0 && approvalsTrustworthy ? String(approvalCount) : undefined;
          return (
            <button aria-current={activeTab === item.key ? "page" : undefined} className={activeTab === item.key ? "active" : ""} key={item.key} type="button" onClick={() => setActiveTab(item.key)}>
              <span>{item.label}</span>
              {badge ? <em aria-label={`${badge} в очереди`}>{badge}</em> : null}
            </button>
          );
        })}
      </nav>

      <CommandPalette isOpen={isPaletteOpen} onClose={() => setPaletteOpen(false)} onNavigate={setActiveTab} />
    </main>
  );
}
