import { useEffect } from "react";
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
  useEffect(() => {
    prepareTelegramViewport();
  }, []);

  const telegram = getTelegramRuntime();
  const displayName = telegram.user?.username
    ? `@${telegram.user.username}`
    : telegram.user?.first_name ?? "Browser preview";

  return (
    <main className="shell" data-mode={telegram.colorScheme}>
      <section className="hero panel">
        <div>
          <p className="eyebrow">Hermes Telegram Mini App · Milestone 1</p>
          <h1>Control Deck</h1>
          <p className="hero-copy">
            Local/browser preview with mock data only. No sidecar, no tunnel, no launchd,
            no privileged actions.
          </p>
        </div>
        <div className="identity-card">
          <span>{telegram.isTelegram ? "Telegram runtime" : "Mock runtime"}</span>
          <strong>{displayName}</strong>
          <small>{telegram.platform}</small>
        </div>
      </section>

      <section className="grid status-grid" aria-label="Hermes status preview">
        {statusCards.map((card) => (
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
