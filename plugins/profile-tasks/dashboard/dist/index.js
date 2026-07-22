(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const REG = window.__HERMES_PLUGINS__;
  if (!SDK || !REG) return;

  const { React } = SDK;
  const h = React.createElement;
  const { useCallback, useEffect, useMemo, useState } = SDK.hooks;
  const C = SDK.components || {};
  const Card = C.Card || "div";
  const CardContent = C.CardContent || "div";
  const Button = C.Button || "button";
  const Badge = C.Badge || "span";
  const Label = C.Label || "label";
  const Select = C.Select || "select";
  const SelectOption = C.SelectOption || "option";
  const cn = (SDK.utils && SDK.utils.cn) || function () { return Array.prototype.slice.call(arguments).filter(Boolean).join(" "); };

  const API = "/api/plugins/profile-tasks";
  const COLUMNS = [
    ["running", "Running"],
    ["blocked", "Blocked"],
    ["review", "Review"],
    ["recent_done", "Recent done"],
    ["ready", "Ready"],
  ];

  function selectChangeHandler(setter) {
    return {
      onValueChange: function (v) { setter(v); },
      onChange: function (e) { setter(e && e.target ? e.target.value : e); },
    };
  }

  function apiError(err) {
    const raw = err && err.message ? String(err.message) : String(err || "Unknown error");
    const m = raw.match(/^(\d{3}):\s*(.*)$/s);
    if (!m) return raw;
    try {
      const parsed = JSON.parse(m[2]);
      return parsed.detail || raw;
    } catch (_) {
      return m[2] || raw;
    }
  }

  function TaskCard(props) {
    const t = props.task;
    const warningCount = (t.warnings || []).length;
    return h("div", { className: "pt-card" },
      h("div", { className: "pt-card-title" }, t.title || t.id),
      h("div", { className: "pt-card-meta" },
        h(Badge, { className: "pt-badge" }, t.id),
        t.priority ? h(Badge, { className: "pt-badge" }, "P" + t.priority) : null,
        t.tenant ? h(Badge, { className: "pt-badge" }, t.tenant) : null,
        warningCount ? h(Badge, { className: "pt-badge pt-warning" }, warningCount + " warning" + (warningCount === 1 ? "" : "s")) : null
      ),
      t.summary_preview ? h("p", { className: "pt-summary" }, t.summary_preview) : null,
      t.latest_run ? h("div", { className: "pt-run" }, "Latest run: ", t.latest_run.status || "unknown", t.latest_run.outcome ? " / " + t.latest_run.outcome : "") : null,
      warningCount ? h("ul", { className: "pt-warnings" }, (t.warnings || []).map(function (w) {
        return h("li", { key: w.kind }, w.message || w.kind);
      })) : null
    );
  }

  function Column(props) {
    const tasks = props.tasks || [];
    return h("section", { className: "pt-column" },
      h("div", { className: "pt-column-head" },
        h("h3", null, props.label),
        h(Badge, { className: "pt-badge" }, String(tasks.length))
      ),
      tasks.length ? tasks.map(function (task) {
        return h(TaskCard, { key: task.id, task: task });
      }) : h("div", { className: "pt-empty" }, "No tasks")
    );
  }

  function ProfileTasks() {
    const [profiles, setProfiles] = useState([]);
    const [boards, setBoards] = useState([]);
    const [profile, setProfile] = useState("");
    const [board, setBoard] = useState("default");
    const [includeReady, setIncludeReady] = useState(false);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const loadMetadata = useCallback(function () {
      return Promise.all([
        SDK.fetchJSON(API + "/profiles"),
        SDK.fetchJSON(API + "/boards"),
      ]).then(function (results) {
        const ps = results[0].profiles || [];
        const bs = results[1].boards || [];
        setProfiles(ps);
        setBoards(bs);
        setProfile(function (current) { return current || (ps[0] && ps[0].name) || ""; });
        setBoard(function (current) { return current || (bs[0] && bs[0].slug) || "default"; });
      });
    }, []);

    const loadTasks = useCallback(function () {
      if (!profile) return Promise.resolve();
      setLoading(true);
      setError(null);
      const qs = new URLSearchParams({ profile: profile, board: board || "default", include_ready: includeReady ? "true" : "false" });
      return SDK.fetchJSON(API + "/tasks?" + qs.toString())
        .then(function (payload) { setData(payload); })
        .catch(function (err) { setError(apiError(err)); })
        .finally(function () { setLoading(false); });
    }, [profile, board, includeReady]);

    useEffect(function () {
      loadMetadata().catch(function (err) { setError(apiError(err)); });
    }, [loadMetadata]);

    useEffect(function () {
      loadTasks();
      const timer = window.setInterval(loadTasks, 15000);
      return function () { window.clearInterval(timer); };
    }, [loadTasks]);

    const activeProfile = useMemo(function () {
      return profiles.find(function (p) { return p.name === profile; }) || null;
    }, [profiles, profile]);

    return h("div", { className: "pt-root" },
      h("div", { className: "pt-header" },
        h("div", null,
          h("h1", null, "Profile Tasks"),
          h("p", null, "Read-only Kanban-by-profile view. Refreshes every 15 seconds; no task mutations are available here.")
        ),
        h(Button, { onClick: loadTasks, disabled: loading || !profile }, loading ? "Refreshing…" : "Refresh")
      ),
      h(Card, { className: "pt-controls" },
        h(CardContent, { className: "pt-controls-inner" },
          h("div", { className: "pt-field" },
            h(Label, null, "Profile"),
            h(Select, Object.assign({ value: profile, className: "pt-select" }, selectChangeHandler(setProfile)),
              profiles.map(function (p) { return h(SelectOption, { key: p.name, value: p.name }, p.name); })
            )
          ),
          h("div", { className: "pt-field" },
            h(Label, null, "Board"),
            h(Select, Object.assign({ value: board, className: "pt-select" }, selectChangeHandler(setBoard)),
              boards.map(function (b) { return h(SelectOption, { key: b.slug, value: b.slug }, b.name || b.slug); })
            )
          ),
          h("label", { className: "pt-check" },
            h("input", { type: "checkbox", checked: includeReady, onChange: function (e) { setIncludeReady(e.target.checked); } }),
            " Include assigned ready tasks"
          )
        )
      ),
      activeProfile ? h("div", { className: "pt-profile-meta" },
        h(Badge, { className: "pt-badge" }, activeProfile.gateway_running ? "gateway running" : "gateway stopped"),
        activeProfile.model ? h(Badge, { className: "pt-badge" }, activeProfile.provider ? activeProfile.provider + ":" + activeProfile.model : activeProfile.model) : null,
        h(Badge, { className: "pt-badge" }, (activeProfile.skill_count || 0) + " skills"),
        activeProfile.description ? h("span", null, activeProfile.description) : null
      ) : null,
      error ? h("div", { className: "pt-error" }, error) : null,
      data && data.warnings && data.warnings.length ? h("div", { className: "pt-warning-box" }, data.warnings.map(function (w) { return h("div", { key: w.kind }, w.message || w.kind); })) : null,
      h("div", { className: cn("pt-board", includeReady && "pt-board-ready") },
        COLUMNS.filter(function (pair) { return pair[0] !== "ready" || includeReady; }).map(function (pair) {
          return h(Column, { key: pair[0], label: pair[1], tasks: data && data.columns ? data.columns[pair[0]] : [] });
        })
      )
    );
  }

  REG.register("profile-tasks", ProfileTasks);
})();
