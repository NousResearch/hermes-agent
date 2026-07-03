import { logLevelFilters, logLineKey, type LogLevelFilter } from "../appModel";
import { type LogLine } from "../mockData";
import { EmptyState } from "./chrome";

export function LogsSection({
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
                <button className={`log-line tap level-${line.level}`} data-selected={key === selectedKey} aria-pressed={key === selectedKey} key={key} type="button" onClick={() => onSelect(line)}>
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
