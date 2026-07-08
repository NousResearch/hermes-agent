import type { ReactNode } from "react";
import { getStatus, getSystemStats } from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

function fmtBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 && u > 0 ? 1 : 0)} ${units[u]}`;
}

function fmtUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return [d && `${d}d`, h && `${h}h`, `${m}m`].filter(Boolean).join(" ");
}

export default function SystemPage() {
  const status = useResource(getStatus);
  const stats = useResource(getSystemStats);

  return (
    <ManagementPage
      title="System"
      actions={
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          onClick={() => {
            status.reload();
            stats.reload();
          }}
        >
          Refresh
        </button>
      }
    >
      <div className="ht-cards">
        <ResourceView resource={status}>
          {(s) => (
            <section className="ht-card">
              <h2 className="ht-card__title">Gateway</h2>
              <dl className="ht-kv">
                <Row k="Version" v={`${s.version} (${s.release_date})`} />
                <Row
                  k="Gateway"
                  v={
                    <span className={s.gateway_running ? "ht-ok" : "ht-dim"}>
                      {s.gateway_running ? `running · pid ${s.gateway_pid ?? "?"}` : "stopped"}
                    </span>
                  }
                />
                <Row k="Active sessions" v={String(s.active_sessions)} />
                <Row
                  k="Auth"
                  v={s.auth_required ? "gated" : "loopback"}
                />
                <Row k="Home" v={<code>{s.hermes_home}</code>} />
                <Row k="Config" v={<code>{s.config_path}</code>} />
                {s.config_version < s.latest_config_version && (
                  <Row
                    k="Config version"
                    v={
                      <span className="ht-warn">
                        {s.config_version} → {s.latest_config_version} (migration available)
                      </span>
                    }
                  />
                )}
              </dl>
              {Object.keys(s.gateway_platforms).length > 0 && (
                <div className="ht-chips">
                  {Object.entries(s.gateway_platforms).map(([name, p]) => (
                    <span
                      key={name}
                      className={`ht-chip${p.connected ? " ht-chip--ok" : ""}`}
                    >
                      {name}
                    </span>
                  ))}
                </div>
              )}
            </section>
          )}
        </ResourceView>

        <ResourceView resource={stats}>
          {(st) => (
            <section className="ht-card">
              <h2 className="ht-card__title">Host</h2>
              <dl className="ht-kv">
                <Row k="OS" v={`${st.os} ${st.os_release} · ${st.arch}`} />
                <Row k="Host" v={st.hostname} />
                <Row k="Python" v={st.python_version} />
                <Row k="CPUs" v={st.cpu_count != null ? String(st.cpu_count) : "—"} />
                {st.cpu_percent != null && <Row k="CPU" v={`${st.cpu_percent.toFixed(0)}%`} />}
                {st.uptime_seconds != null && (
                  <Row k="Uptime" v={fmtUptime(st.uptime_seconds)} />
                )}
                {st.memory && (
                  <Row
                    k="Memory"
                    v={`${fmtBytes(st.memory.used)} / ${fmtBytes(st.memory.total)} (${st.memory.percent.toFixed(0)}%)`}
                  />
                )}
                {st.disk && (
                  <Row
                    k="Disk"
                    v={`${fmtBytes(st.disk.used)} / ${fmtBytes(st.disk.total)} (${st.disk.percent.toFixed(0)}%)`}
                  />
                )}
                {!st.psutil && <Row k="Note" v={<span className="ht-dim">psutil not installed — limited stats</span>} />}
              </dl>
            </section>
          )}
        </ResourceView>
      </div>
    </ManagementPage>
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
