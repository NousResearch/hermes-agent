import { useState, type ReactNode } from "react";
import {
  getAnalytics,
  getModelsAnalytics,
  type AnalyticsDailyEntry,
} from "@/api/analytics";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { fmtCost, fmtNumber, relTime } from "./mgmtFormat";

const DAY_OPTIONS = [7, 30, 90] as const;

export default function AnalyticsPage() {
  const [days, setDays] = useState<number>(30);
  const usage = useResource(() => getAnalytics(days), [days]);
  const models = useResource(() => getModelsAnalytics(days), [days]);

  return (
    <ManagementPage
      title="Analytics"
      actions={
        <>
          <select
            className="ht-select"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            aria-label="Time range"
          >
            {DAY_OPTIONS.map((d) => (
              <option key={d} value={d}>
                Last {d} days
              </option>
            ))}
          </select>
          <button
            type="button"
            className="ht-btn ht-btn--sm"
            onClick={() => {
              usage.reload();
              models.reload();
            }}
          >
            Refresh
          </button>
        </>
      }
    >
      <div className="ht-cards">
        <ResourceView resource={usage}>
          {(d) => (
            <section className="ht-card">
              <h2 className="ht-card__title">Totals</h2>
              <dl className="ht-kv">
                <Row k="Input tokens" v={fmtNumber(d.totals.total_input)} />
                <Row k="Output tokens" v={fmtNumber(d.totals.total_output)} />
                <Row k="Cache read tokens" v={fmtNumber(d.totals.total_cache_read)} />
                <Row k="Reasoning tokens" v={fmtNumber(d.totals.total_reasoning)} />
                <Row k="Sessions" v={fmtNumber(d.totals.total_sessions)} />
                <Row k="API calls" v={fmtNumber(d.totals.total_api_calls)} />
                <Row k="Estimated cost" v={fmtCost(d.totals.total_estimated_cost)} />
                <Row k="Actual cost" v={fmtCost(d.totals.total_actual_cost)} />
              </dl>
            </section>
          )}
        </ResourceView>

        <ResourceView resource={usage}>
          {(d) => (
            <section className="ht-card">
              <h2 className="ht-card__title">Tokens per day</h2>
              {d.daily.length === 0 ? (
                <p className="ht-muted">No usage in this range.</p>
              ) : (
                <TokenBars daily={d.daily} />
              )}
            </section>
          )}
        </ResourceView>
      </div>

      <ResourceView resource={models} empty={(d) => d.models.length === 0}>
        {(d) => (
          <section className="ht-card">
            <h2 className="ht-card__title">By model</h2>
            <table className="ht-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Input</th>
                  <th>Output</th>
                  <th>Cost</th>
                  <th>Sessions</th>
                  <th>Last used</th>
                </tr>
              </thead>
              <tbody>
                {d.models.map((m) => (
                  <tr key={`${m.provider}:${m.model}`}>
                    <td>
                      <code>{m.model}</code>
                    </td>
                    <td>
                      <span className="ht-chip">{m.provider}</span>
                    </td>
                    <td>{fmtNumber(m.input_tokens)}</td>
                    <td>{fmtNumber(m.output_tokens)}</td>
                    <td>{fmtCost(m.estimated_cost)}</td>
                    <td>{fmtNumber(m.sessions)}</td>
                    <td className="ht-sm ht-muted">{relTime(m.last_used_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

// Dependency-free horizontal bar chart of input+output tokens per day, drawn
// with plain divs. Each row is a labelled progress-style bar; the numeric value
// stays visible and aria-label carries the full figure for screen readers.
function TokenBars({ daily }: { daily: AnalyticsDailyEntry[] }) {
  const totals = daily.map((d) => d.input_tokens + d.output_tokens);
  const max = Math.max(1, ...totals);
  return (
    <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
      {daily.map((d, i) => {
        const value = totals[i];
        const pct = Math.round((value / max) * 100);
        return (
          <li
            key={d.day}
            style={{ display: "flex", alignItems: "center", gap: "0.5rem", margin: "0.15rem 0" }}
          >
            <span className="ht-sm ht-mono ht-muted" style={{ width: "5.5rem", flex: "0 0 auto" }}>
              {d.day}
            </span>
            <span
              role="img"
              aria-label={`${d.day}: ${value.toLocaleString()} tokens`}
              style={{
                flex: 1,
                background: "var(--ht-bar-track, rgba(127,127,127,0.15))",
                borderRadius: "3px",
                overflow: "hidden",
              }}
            >
              <span
                style={{
                  display: "block",
                  width: `${pct}%`,
                  minWidth: value > 0 ? "2px" : 0,
                  height: "0.9rem",
                  background: "var(--ht-bar-fill, currentColor)",
                  opacity: 0.65,
                }}
              />
            </span>
            <span className="ht-sm ht-mono" style={{ width: "4rem", flex: "0 0 auto", textAlign: "right" }}>
              {fmtNumber(value)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Row({ k, v }: { k: string; v: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}
