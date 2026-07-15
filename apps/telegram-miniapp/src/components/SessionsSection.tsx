import { type SessionPreview } from "../mockData";
import { EmptyState } from "./chrome";

export function SessionsSection({ sessions, selectedId, onSelect }: { sessions: SessionPreview[]; selectedId: string; onSelect: (session: SessionPreview) => void }) {
  if (sessions.length === 0) {
    return <EmptyState title="Сессий нет" text="Сервер ответил пустым списком. Локальные mock-данные не подмешиваются после успешного обновления." />;
  }

  const selected = sessions.find((session) => session.id === selectedId) ?? sessions[0];

  return (
    <section className="drilldown-workspace" aria-label="Сессии агентов">
      <div className="stack-list compact-stack">
        {sessions.map((session) => (
          <button className={`list-card glass-card tap tone-${session.tone}`} data-selected={session.id === selected.id} aria-pressed={session.id === selected.id} key={session.id} type="button" onClick={() => onSelect(session)}>
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
