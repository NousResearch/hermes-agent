import { useEffect, useMemo, useState } from "react";
import { authenticateTelegram, fetchStatusSnapshot, hasMiniAppApi, type StatusSnapshot } from "./api";
import { recentLogs, statusCards, quickActions, type QuickAction } from "./mockData";
import { getTelegramRuntime, prepareTelegramViewport } from "./telegram";

function RiskBadge({ action }: { action: QuickAction }) {
  const label = action.risk === "read_only" ? "read-only" : action.risk;
  return <span className={`risk risk-${action.risk}`}>{label}</span>;
}

function CommandPreview({ command }: { command: string }) {
  const visible = command.startsWith("mock:") || command.startsWith("disabled:") ? command : `/${command}`;
  return <code>{visible}</code>;
}

export function App() {
  const [snapshot, setSnapshot] = useState<StatusSnapshot | null>(null);
  const [apiState, setApiState] = useState<"mock" | "connecting" | "connected" | "offline">("mock");

  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const apiConfigured = hasMiniAppApi();

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
    : telegram.user?.first_name ?? "Browser preview";

  const cards = useMemo(() => {
    if (!snapshot) {
      return statusCards;
    }
    return [
      {
        label: "Gateway",
        value: snapshot.gateway.running ? "Online" : "Offline",
        tone: snapshot.gateway.running ? "ok" : "warn",
      },
      { label: "State", value: snapshot.gateway.state, tone: snapshot.gateway.busy ? "warn" : "ok" },
      { label: "Agents", value: String(snapshot.gateway.active_agents), tone: snapshot.gateway.busy ? "warn" : "ok" },
      { label: "Actions", value: snapshot.miniapp.actions_enabled ? "Enabled" : "Disabled", tone: "warn" },
    ] as const;
  }, [snapshot]);

  const connectionLabel =
    apiState === "connected"
      ? "real read-only status"
      : apiState === "connecting"
        ? "connecting to sidecar"
        : apiState === "offline"
          ? "sidecar unavailable — mock fallback"
          : apiConfigured
            ? "API configured — waiting for Telegram initData"
            : "mock/local preview";

  return (
    <main className="shell" data-mode={telegram.colorScheme}>
      <section className="hero panel">
        <div>
          <p className="eyebrow">Hermes Telegram Mini App · Milestone 2</p>
          <h1>Control Deck</h1>
          <p className="hero-copy">
            Read-only status integration with mock fallback. No tunnel, no launchd,
            no privileged actions, no stored Telegram initData.
          </p>
          <div className={`connection-banner state-${apiState}`}>{connectionLabel}</div>
        </div>
        <div className="identity-card">
          <span>{telegram.isTelegram ? "Telegram runtime" : "Mock runtime"}</span>
          <strong>{displayName}</strong>
          <small>{telegram.platform}</small>
        </div>
      </section>

      <section className="grid status-grid" aria-label="Hermes status preview">
        {cards.map((card) => (
          <article className={`panel status-card tone-${card.tone}`} key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
          </article>
        ))}
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Quick actions</p>
            <h2>Safe by default</h2>
          </div>
          <span className="lock-pill">actions disabled</span>
        </div>
        <div className="action-list">
          {quickActions.map((action) => (
            <article className="action-row" key={action.command}>
              <div>
                <div className="action-title">
                  <strong>{action.label}</strong>
                  <RiskBadge action={action} />
                </div>
                <p>{action.description}</p>
              </div>
              <CommandPreview command={action.command} />
            </article>
          ))}
        </div>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-heading compact">
            <h2>Approvals</h2>
            <span className="risk risk-read_only">empty</span>
          </div>
          <p className="muted">
            The approval queue is intentionally empty in this milestone. Real action requests
            require the server-side approve-gate state machine from Milestone 4.
          </p>
        </article>

        <article className="panel">
          <div className="section-heading compact">
            <h2>Recent logs</h2>
            <span className="risk risk-disabled">redacted mock</span>
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
    </main>
  );
}
