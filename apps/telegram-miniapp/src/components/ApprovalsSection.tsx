import { type ApprovalPreview } from "../mockData";
import { EmptyState, RiskBadge } from "./chrome";

function DecisionReadiness({ approval }: { approval: ApprovalPreview }) {
  const guardrails = [
    {
      label: "Действия сервера",
      value: "не подключены",
      ready: false,
      detail: "В Mini App нет endpoint для approve/reject/restart. Их добавление требует отдельного approved design.",
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
          <p className="mono-label">M12 / PREFLIGHT LOCK</p>
          <h3>Решение требует отдельного design approval</h3>
        </div>
        <span className="lock-pill">read-only</span>
      </div>
      <p>Этот блок заранее показывает, что должно быть проверено перед будущим approve/reject. Сейчас он ничего не отправляет, не меняет и не создаёт action route.</p>
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

export function ApprovalsSection({ approvals, selectedId, onSelect }: { approvals: ApprovalPreview[]; selectedId: string; onSelect: (approval: ApprovalPreview) => void }) {
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
