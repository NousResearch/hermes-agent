(function () {
  var sdk = window.__HERMES_PLUGIN_SDK__;
  var registry = window.__HERMES_PLUGINS__;
  if (!sdk || !registry) return;

  var React = sdk.React;
  var hooks = sdk.hooks;
  var fetchJSON = sdk.fetchJSON;
  var Card = sdk.components.Card;
  var CardHeader = sdk.components.CardHeader;
  var CardTitle = sdk.components.CardTitle;
  var CardContent = sdk.components.CardContent;
  var Button = sdk.components.Button;
  var Badge = sdk.components.Badge;
  var Input = sdk.components.Input;
  var Select = sdk.components.Select;
  var SelectOption = sdk.components.SelectOption;

  function h(type, props) {
    var children = Array.prototype.slice.call(arguments, 2);
    return React.createElement.apply(React, [type, props || {}].concat(children));
  }

  function fmtInt(v) {
    return Number(v || 0).toLocaleString();
  }

  function fmtUsd(v) {
    var n = Number(v || 0);
    if (!Number.isFinite(n)) n = 0;
    return "$" + n.toFixed(n >= 1 ? 2 : 4);
  }

  function fmtTs(v) {
    if (!v) return "—";
    var ms = Number(v) * 1000;
    if (!Number.isFinite(ms)) return "—";
    return new Date(ms).toLocaleString();
  }

  function tokenTotal(row) {
    return Number(row.input_tokens || 0) + Number(row.output_tokens || 0) + Number(row.reasoning_tokens || 0);
  }

  function SummaryCard(props) {
    return h(Card, { className: "project-usage-card" },
      h(CardHeader, null, h(CardTitle, null, props.label)),
      h(CardContent, null,
        h("div", { className: "project-usage-metric" }, props.value),
        props.sub ? h("div", { className: "project-usage-muted" }, props.sub) : null
      )
    );
  }

  function ProjectUsagePage() {
    var _useState = hooks.useState(null), data = _useState[0], setData = _useState[1];
    var _useState2 = hooks.useState(""), error = _useState2[0], setError = _useState2[1];
    var _useState3 = hooks.useState(true), loading = _useState3[0], setLoading = _useState3[1];
    var _useState4 = hooks.useState(""), board = _useState4[0], setBoard = _useState4[1];
    var _useState5 = hooks.useState(""), taskId = _useState5[0], setTaskId = _useState5[1];
    var _useState6 = hooks.useState(""), search = _useState6[0], setSearch = _useState6[1];

    var load = hooks.useCallback(function (opts) {
      opts = opts || {};
      setLoading(true);
      setError("");
      var params = new URLSearchParams();
      if (board) params.set("board", board);
      if (taskId) params.set("task_id", taskId);
      params.set("refresh", opts.refresh === false ? "false" : "true");
      fetchJSON("/api/plugins/project_usage/summary?" + params.toString())
        .then(function (json) { setData(json); })
        .catch(function (err) { setError(String(err && err.message ? err.message : err)); })
        .finally(function () { setLoading(false); });
    }, [board, taskId]);

    hooks.useEffect(function () { load({ refresh: true }); }, [load]);

    var boards = data && data.boards ? data.boards : [];
    var tasks = data && data.tasks ? data.tasks : [];
    var runs = data && data.runs ? data.runs : [];
    var totals = data && data.totals ? data.totals : {};
    var query = search.trim().toLowerCase();
    var visibleTasks = tasks.filter(function (t) {
      if (!query) return true;
      return String(t.task_id || "").toLowerCase().includes(query) ||
        String(t.task_title || "").toLowerCase().includes(query) ||
        String(t.board_slug || "").toLowerCase().includes(query);
    });

    return h("div", { className: "project-usage" },
      h("div", { className: "project-usage-header" },
        h("div", null,
          h("h1", null, "Project Usage"),
          h("p", null, "Usage ledger under Hermes home, backfilled from state.db and Kanban board metadata.")
        ),
        h("div", { className: "project-usage-actions" },
          h(Button, { onClick: function () { load({ refresh: true }); }, disabled: loading }, loading ? "Refreshing…" : "Refresh ledger")
        )
      ),

      error ? h("div", { className: "project-usage-error" }, error) : null,

      h("div", { className: "project-usage-grid" },
        h(SummaryCard, { label: "Estimated cost", value: fmtUsd(totals.estimated_cost_usd), sub: "actual " + fmtUsd(totals.actual_cost_usd) }),
        h(SummaryCard, { label: "Input tokens", value: fmtInt(totals.input_tokens), sub: "cache read " + fmtInt(totals.cache_read_tokens) }),
        h(SummaryCard, { label: "Output tokens", value: fmtInt(totals.output_tokens), sub: "reasoning " + fmtInt(totals.reasoning_tokens) }),
        h(SummaryCard, { label: "Coverage", value: fmtInt(totals.tasks) + " tasks", sub: fmtInt(totals.sessions) + " sessions / " + fmtInt(totals.entries) + " entries" })
      ),

      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Filters")),
        h(CardContent, null,
          h("div", { className: "project-usage-filters" },
            h("label", null, "Board",
              h(Select, {
                value: board || "__all__",
                onValueChange: function (v) { setTaskId(""); setBoard(v === "__all__" ? "" : v); },
                onChange: function (e) { var v = e.target.value; setTaskId(""); setBoard(v === "__all__" ? "" : v); }
              },
                h(SelectOption, { value: "__all__" }, "All boards"),
                boards.filter(function (b) { return b.board_slug !== "__unassigned__"; }).map(function (b) {
                  return h(SelectOption, { key: b.board_slug, value: b.board_slug }, b.board_name + " (" + b.board_slug + ")");
                })
              )
            ),
            h("label", null, "Search tasks",
              h(Input, { value: search, placeholder: "task id/title/board", onChange: function (e) { setSearch(e.target.value); } })
            ),
            taskId ? h("div", { className: "project-usage-selected" },
              h(Badge, null, "Drilldown: " + taskId),
              h(Button, { onClick: function () { setTaskId(""); } }, "Clear")
            ) : null
          )
        )
      ),

      h("div", { className: "project-usage-columns" },
        h(Card, null,
          h(CardHeader, null, h(CardTitle, null, "Per-board totals")),
          h(CardContent, null,
            h("div", { className: "project-usage-table-wrap" },
              h("table", { className: "project-usage-table" },
                h("thead", null, h("tr", null,
                  h("th", null, "Board"), h("th", null, "Tasks"), h("th", null, "Sessions"),
                  h("th", null, "Tokens"), h("th", null, "Est. cost"), h("th", null, "Latest")
                )),
                h("tbody", null,
                  boards.map(function (b) {
                    return h("tr", { key: b.board_slug, onClick: function () { if (b.board_slug !== "__unassigned__") { setTaskId(""); setBoard(b.board_slug); } } },
                      h("td", null, h("strong", null, b.board_name), h("div", { className: "project-usage-muted" }, b.board_slug)),
                      h("td", null, fmtInt(b.tasks)),
                      h("td", null, fmtInt(b.sessions)),
                      h("td", null, fmtInt(tokenTotal(b))),
                      h("td", null, fmtUsd(b.estimated_cost_usd)),
                      h("td", null, fmtTs(b.latest_started_at))
                    );
                  })
                )
              )
            )
          )
        ),

        h(Card, null,
          h(CardHeader, null, h(CardTitle, null, "Per-task drilldown")),
          h(CardContent, null,
            h("div", { className: "project-usage-table-wrap" },
              h("table", { className: "project-usage-table" },
                h("thead", null, h("tr", null,
                  h("th", null, "Task"), h("th", null, "Board"), h("th", null, "Runs"),
                  h("th", null, "Tokens"), h("th", null, "Est. cost"), h("th", null, "Last ended")
                )),
                h("tbody", null,
                  visibleTasks.map(function (t) {
                    return h("tr", { key: t.board_slug + ":" + t.task_id, onClick: function () { setBoard(t.board_slug || ""); setTaskId(t.task_id); } },
                      h("td", null, h("strong", null, t.task_title || t.task_id), h("div", { className: "project-usage-muted" }, t.task_id + " · " + (t.task_status || "unknown"))),
                      h("td", null, t.board_name || t.board_slug || "—"),
                      h("td", null, fmtInt(t.runs)),
                      h("td", null, fmtInt(tokenTotal(t))),
                      h("td", null, fmtUsd(t.estimated_cost_usd)),
                      h("td", null, fmtTs(t.last_ended_at))
                    );
                  })
                )
              )
            )
          )
        )
      ),

      taskId ? h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Run/session rows for " + taskId)),
        h(CardContent, null,
          h("div", { className: "project-usage-table-wrap" },
            h("table", { className: "project-usage-table" },
              h("thead", null, h("tr", null,
                h("th", null, "Run"), h("th", null, "Session"), h("th", null, "Model"),
                h("th", null, "Tokens"), h("th", null, "Est. cost"), h("th", null, "Status")
              )),
              h("tbody", null,
                runs.map(function (r) {
                  return h("tr", { key: r.source_id },
                    h("td", null, r.run_id || "—"),
                    h("td", null, h("div", null, r.session_title || r.session_id || "unlinked"), h("div", { className: "project-usage-muted" }, r.session_id || "—")),
                    h("td", null, r.model || "—"),
                    h("td", null, fmtInt(tokenTotal(r))),
                    h("td", null, fmtUsd(r.estimated_cost_usd)),
                    h("td", null, (r.run_status || "—") + (r.run_outcome ? " / " + r.run_outcome : ""))
                  );
                })
              )
            )
          )
        )
      ) : null,

      data ? h("div", { className: "project-usage-footer" },
        "Ledger: " + data.ledger_path + " · Last backfill: " + fmtTs(data.last_backfill_at)
      ) : null
    );
  }

  registry.register("project_usage", ProjectUsagePage);
})();
