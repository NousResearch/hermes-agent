import { type CSSProperties } from "react";
import { type CapabilityItem } from "../api";
import { type EndpointHealthItem } from "../appModel";
import { quickActions } from "../mockData";
import { RiskBadge } from "./chrome";

export function StatusSection({
  cards,
  safetyText,
  approvalCount,
  capabilities,
  endpointHealth,
}: {
  cards: ReadonlyArray<{ label: string; value: string; meta: string; tone: string }>;
  safetyText: string;
  approvalCount: number;
  capabilities: CapabilityItem[];
  endpointHealth: EndpointHealthItem[];
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

      {capabilities.length > 0 ? (
        <section className="capability-card glass-card" aria-label="Матрица возможностей">
          <div className="section-heading">
            <div>
              <p className="mono-label">M13 / CAPABILITIES</p>
              <h2>Что разрешено сейчас</h2>
            </div>
            <span className="lock-pill">по данным сервера</span>
          </div>
          <div className="capability-grid">
            {capabilities.map((capability) => (
              <article className="capability-item" data-enabled={capability.enabled} key={capability.id}>
                <span>{capability.enabled ? "разрешено" : "заблокировано"}</span>
                <strong>{capability.label}</strong>
                <p>{capability.reason}</p>
                <em>{capability.mode}</em>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <section className="endpoint-health-card glass-card" aria-label="Состояние снимков данных">
        <div className="section-heading">
          <div>
            <p className="mono-label">M14 / ENDPOINT HEALTH</p>
            <h2>Снимки данных</h2>
          </div>
          <span className="lock-pill">только статус</span>
        </div>
        <div className="endpoint-health-list">
          {endpointHealth.map((item) => (
            <article className="endpoint-health-row" data-state={item.state} key={item.key}>
              <span>{item.label}</span>
              <strong>{item.state === "ok" ? "готов" : item.state === "checking" ? "проверка" : item.state === "degraded" ? "деградация" : "превью"}</strong>
              <small>{item.detail}</small>
            </article>
          ))}
        </div>
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
            // Informational tiles: no onClick, no tap affordance (they are not
            // controls — see styles.css .quick-tile which drops pointer/hover).
            <article className="quick-tile" key={action.id} style={{ "--delay": `${index * 45}ms` } as CSSProperties}>
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
