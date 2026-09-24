(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const h = React.createElement;
  const hooks = SDK.hooks || {};
  const useEffect = hooks.useEffect;
  const useState = hooks.useState;
  const components = SDK.components || {};
  const Card = components.Card || "div";
  const CardHeader = components.CardHeader || "div";
  const CardTitle = components.CardTitle || "h2";
  const CardContent = components.CardContent || "div";
  const Badge = components.Badge || "span";
  const Button = components.Button || "button";

  const API = "/api/plugins/agent-roster";

  function badgeClass(severity) {
    if (severity === "high" || severity === "critical") return "agent-roster-badge-danger";
    if (severity === "medium") return "agent-roster-badge-warning";
    return "agent-roster-badge-muted";
  }

  function Stat(props) {
    return h("div", { className: "agent-roster-stat" },
      h("div", { className: "agent-roster-stat-value" }, String(props.value == null ? "—" : props.value)),
      h("div", { className: "agent-roster-stat-label" }, props.label)
    );
  }

  function AgentRosterPage() {
    const _state = useState(null);
    const data = _state[0];
    const setData = _state[1];
    const _err = useState(null);
    const error = _err[0];
    const setError = _err[1];
    const _loading = useState(false);
    const loading = _loading[0];
    const setLoading = _loading[1];

    function load() {
      setLoading(true);
      setError(null);
      SDK.fetchJSON(API + "/roster")
        .then(function (payload) { setData(payload); })
        .catch(function (err) { setError(String(err && err.message ? err.message : err)); })
        .finally(function () { setLoading(false); });
    }

    useEffect(function () { load(); }, []);

    if (error) {
      return h("div", { className: "agent-roster-page" },
        h(Card, null,
          h(CardHeader, null, h(CardTitle, null, "Agent Roster")),
          h(CardContent, null,
            h("p", { className: "agent-roster-error" }, error),
            h(Button, { onClick: load }, "Retry")
          )
        )
      );
    }

    if (!data) {
      return h("div", { className: "agent-roster-page" }, "Loading Agent Roster…");
    }

    const summary = data.summary || {};
    const profiles = data.profiles || [];
    const violations = data.violations || [];
    const stages = ((data.pipeline || {}).stages || []);

    return h("div", { className: "agent-roster-page" },
      h("div", { className: "agent-roster-header" },
        h("div", null,
          h("h1", null, "Agent Roster"),
          h("p", null, "Profile roles, Kanban drift, PM/Reviewer gates, and policy enforcement.")
        ),
        h(Button, { onClick: load, disabled: loading }, loading ? "Refreshing…" : "Refresh")
      ),
      h("div", { className: "agent-roster-stats" },
        h(Stat, { label: "Profiles", value: summary.profile_count }),
        h(Stat, { label: "Roles", value: summary.role_count }),
        h(Stat, { label: "Boards", value: summary.board_count }),
        h(Stat, { label: "Tasks", value: summary.task_count }),
        h(Stat, { label: "Violations", value: summary.violation_count })
      ),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Pipeline")),
        h(CardContent, null,
          h("div", { className: "agent-roster-pipeline" }, stages.map(function (stage) {
            return h("div", { className: "agent-roster-stage", key: stage.key },
              h("strong", null, stage.label || stage.key),
              h("span", null, String(stage.task_count || 0) + " tasks"),
              h("small", null, "ready " + (stage.ready_count || 0) + " · running " + (stage.running_count || 0) + " · blocked " + (stage.blocked_count || 0))
            );
          }))
        )
      ),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Profiles")),
        h(CardContent, null,
          h("div", { className: "agent-roster-grid" }, profiles.map(function (profile) {
            const role = profile.role || {};
            return h("div", { className: "agent-roster-profile", key: profile.name },
              h("div", { className: "agent-roster-profile-top" },
                h("strong", null, profile.name),
                h(Badge, null, role.status || "missing")
              ),
              h("p", null, role.mission || "No profile_role metadata."),
              h("small", null, "Allowed: " + ((role.allowed_task_types || []).join(", ") || "—")),
              h("small", null, "Forbidden: " + ((role.forbidden || []).join(", ") || "—"))
            );
          }))
        )
      ),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Violations")),
        h(CardContent, null,
          violations.length === 0
            ? h("p", null, "No drift detected.")
            : h("div", { className: "agent-roster-violations" }, violations.slice(0, 50).map(function (violation, idx) {
                return h("div", { className: "agent-roster-violation", key: violation.code + String(idx) },
                  h("span", { className: badgeClass(violation.severity) }, violation.severity || "info"),
                  h("div", null,
                    h("strong", null, violation.code),
                    h("p", null, violation.message),
                    violation.task_id ? h("small", null, "Task " + violation.task_id + " · Board " + (violation.board || "default")) : null
                  )
                );
              }))
        )
      )
    );
  }

  window.__HERMES_PLUGINS__.register("agent-roster", AgentRosterPage);
})();
