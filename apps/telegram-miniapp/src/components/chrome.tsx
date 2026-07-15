import { type SnapshotMeta } from "../api";
import { navTitles, refreshLabels, riskLabels, type FreshnessState } from "../appModel";
import { type NavKey, type QuickAction } from "../mockData";

export function RiskBadge({ action }: { action: Pick<QuickAction, "risk"> }) {
  return <span className={`risk risk-${action.risk}`}>{riskLabels[action.risk]}</span>;
}

export function SectionIntro({
  activeTab,
  freshness,
  isRefreshEnabled,
  lastRefreshText,
  onOpenPalette,
  onRefresh,
}: {
  activeTab: NavKey;
  freshness: FreshnessState;
  isRefreshEnabled: boolean;
  lastRefreshText: string;
  onOpenPalette: () => void;
  onRefresh: () => void;
}) {
  return (
    <section className="section-intro glass-card" aria-label="Текущий раздел">
      <div>
        <p className="mono-label">M10 / LIVE READ-ONLY</p>
        <h2>{navTitles[activeTab]}</h2>
        <p>Данные обновляются только чтением. Ручное обновление не запускает команды и не меняет состояние Hermes.</p>
        <div className="freshness-row" data-state={freshness}>
          <span>{refreshLabels[freshness]}</span>
          <small>{lastRefreshText}</small>
        </div>
      </div>
      <div className="section-actions">
        <button className="refresh-button tap" type="button" disabled={!isRefreshEnabled || freshness === "refreshing"} onClick={onRefresh}>
          {freshness === "refreshing" ? "Обновляю" : "Обновить"}
        </button>
        <button className="palette-button tap" type="button" onClick={onOpenPalette}>
          Палитра
        </button>
      </div>
    </section>
  );
}

export function SourceStrip({ meta }: { meta: SnapshotMeta | null }) {
  if (!meta) return null;

  return (
    <div className="source-strip" aria-label="Источник данных">
      <span>{meta.source_label}</span>
      <small>{meta.redaction}</small>
      <em>{meta.contains_live_actions ? "actions linked" : "no actions"}</em>
    </div>
  );
}

export function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <section className="empty-state glass-card" aria-label={title}>
      <p className="mono-label">ПУСТОЙ SERVER SNAPSHOT</p>
      <h2>{title}</h2>
      <p>{text}</p>
    </section>
  );
}
