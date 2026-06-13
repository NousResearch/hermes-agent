(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { useEffect, useState } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;

  function LogMeta(props) {
    const log = props.log || {};
    return h("div", { className: "rounded-md border p-3 text-sm space-y-1" },
      h("div", { className: "font-medium" }, "gateway.log"),
      h("div", { className: "text-xs text-muted-foreground break-all" }, log.path || "—"),
      h("div", null, log.exists ? `present · size ${log.size} · errors counted ${log.recent_error_count_tail_2000}` : "missing")
    );
  }

  function StartupRecoveryPage() {
    const [state, setState] = useState({ loading: true, error: null, data: null });
    function load() {
      setState((s) => Object.assign({}, s, { loading: true, error: null }));
      fetchJSON("/api/plugins/startup-recovery/status")
        .then((data) => setState({ loading: false, error: null, data }))
        .catch((err) => setState({ loading: false, error: String(err && err.message || err), data: null }));
    }
    useEffect(load, []);
    const profiles = (state.data && state.data.profiles) || [];
    return h("div", { className: "space-y-6" },
      h("div", { className: "flex items-start justify-between gap-4" },
        h("div", null,
          h("h1", { className: "text-2xl font-semibold" }, "Startup Recovery"),
          h("p", { className: "text-sm text-muted-foreground" }, "Read-only local gateway health. No Telegram sends, restarts, process kills, or log contents.")
        ),
        h(Button, { onClick: load, disabled: state.loading }, state.loading ? "Refreshing…" : "Refresh")
      ),
      state.error && h(Card, { className: "border-destructive" }, h(CardContent, { className: "pt-6 text-sm text-destructive" }, state.error)),
      state.data && h("div", { className: "flex flex-wrap gap-2" },
        h(Badge, { variant: state.data.overall === "OK" ? "default" : "destructive" }, `Overall: ${state.data.overall}`),
        h(Badge, { variant: state.data.problem_count ? "destructive" : "secondary" }, `${state.data.problem_count} problem(s)`),
        h(Badge, { variant: "outline" }, `Telegram TCP: ${state.data.telegram_tcp_connections_detected == null ? "unknown" : state.data.telegram_tcp_connections_detected}`),
        h(Badge, { variant: "outline" }, state.data.scope)
      ),
      h("div", { className: "grid gap-4 md:grid-cols-2" }, profiles.map((profile) =>
        h(Card, { key: profile.profile },
          h(CardHeader, null,
            h("div", { className: "flex items-start justify-between gap-2" },
              h("div", null,
                h(CardTitle, null, profile.profile),
                h("p", { className: "text-xs text-muted-foreground" }, profile.bot)
              ),
              h("div", { className: "flex gap-2" },
                h(Badge, { variant: profile.tmux_status === "OK" ? "default" : "destructive" }, `tmux ${profile.tmux_status}`),
                h(Badge, { variant: profile.process_status === "OK" ? "default" : "destructive" }, `process ${profile.process_status}`)
              )
            )
          ),
          h(CardContent, { className: "space-y-3" },
            h("div", { className: "text-sm" }, "Matched session: ", h("span", { className: "font-mono" }, profile.matched_session || "—")),
            h("div", { className: "text-xs text-muted-foreground" }, "Expected: ", (profile.expected_sessions || []).join(", ")),
            h(LogMeta, { log: profile.log_file }),
            (function () {
              const stateDb = profile.state_db || {};
              return h("div", { className: "rounded-md border p-3 text-sm" },
                h("div", { className: "font-medium" }, "state.db"),
                h("div", { className: "text-xs text-muted-foreground break-all" }, stateDb.path || "—"),
                h("div", null, stateDb.exists ? `present · size ${stateDb.size}` : "missing")
              );
            })()
          )
        )
      )),
      state.loading && h("p", { className: "text-sm text-muted-foreground" }, "Loading gateway health…")
    );
  }

  Registry.register("startup-recovery", StartupRecoveryPage);
})();
