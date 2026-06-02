/**
 * ClawTeam — Hermes Dashboard Plugin
 *
 * Lists teams discovered via `clawteam team discover` and lets the operator
 * drill into one for member + recent-activity status. Plain IIFE, no build
 * step. Uses window.__HERMES_PLUGIN_SDK__ for React + DS primitives, and
 * SDK.fetchJSON for auth-aware, base-path-aware HTTP.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const REG = window.__HERMES_PLUGINS__;
  if (!SDK || !REG) {
    console.error("[clawteam] Hermes plugin SDK or registry not available on window");
    return;
  }

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { Card, CardContent, Button, Badge } = SDK.components;
  const { useState, useEffect, useCallback } = SDK.hooks;

  const API_BASE = "/api/plugins/clawteam";

  function fetchJson(path) {
    return fetchJSON(API_BASE + path);
  }

  function TeamRow(props) {
    const { team, selected, onSelect } = props;
    const name = (team && (team.name || team.team_name)) || String(team);
    const leader = team && (team.leader || team.leader_agent);
    return h(
      "div",
      {
        className: "ct-row" + (selected ? " ct-row-selected" : ""),
        onClick: function () { onSelect(name); },
      },
      h("span", { className: "ct-name" }, name),
      leader ? h(Badge, { variant: "outline" }, "leader: " + leader) : null
    );
  }

  function TeamDetail(props) {
    const { name } = props;
    const [data, setData] = useState(null);
    const [err, setErr] = useState(null);

    useEffect(function () {
      let cancelled = false;
      setData(null); setErr(null);
      fetchJson("/teams/" + encodeURIComponent(name))
        .then(function (j) { if (!cancelled) setData(j.team); })
        .catch(function (e) { if (!cancelled) setErr(e.message); });
      return function () { cancelled = true; };
    }, [name]);

    if (err) return h("div", { className: "ct-error" }, "Error: " + err);
    if (!data) return h("div", { className: "ct-muted" }, "Loading…");
    return h(
      "pre",
      { className: "ct-detail" },
      JSON.stringify(data, null, 2)
    );
  }

  function App() {
    const [teams, setTeams] = useState([]);
    const [selected, setSelected] = useState(null);
    const [err, setErr] = useState(null);
    const [loading, setLoading] = useState(false);

    // useRef + cancel token: a tab-switch mid-fetch must not setState on
    // unmounted root. Each refresh bumps the token; only the latest one
    // is allowed to land.
    const reqRef = React.useRef(0);
    const mountedRef = React.useRef(true);
    useEffect(function () {
      return function () { mountedRef.current = false; };
    }, []);

    const refresh = useCallback(function () {
      const token = ++reqRef.current;
      setLoading(true); setErr(null);
      fetchJson("/teams")
        .then(function (j) {
          if (!mountedRef.current || token !== reqRef.current) return;
          setTeams(j.teams || []);
        })
        .catch(function (e) {
          if (!mountedRef.current || token !== reqRef.current) return;
          setErr(e.message);
        })
        .finally(function () {
          if (!mountedRef.current || token !== reqRef.current) return;
          setLoading(false);
        });
    }, []);

    useEffect(function () { refresh(); }, [refresh]);

    return h(
      "div",
      { className: "ct-root" },
      h(
        "div",
        { className: "ct-header" },
        h("h2", null, "ClawTeam"),
        h(Button, { onClick: refresh, disabled: loading }, loading ? "…" : "Refresh")
      ),
      err ? h("div", { className: "ct-error" }, "Error: " + err) : null,
      h(
        "div",
        { className: "ct-cols" },
        h(
          Card,
          { className: "ct-list" },
          h(
            CardContent,
            null,
            teams.length === 0 && !loading
              ? h("div", { className: "ct-muted" }, "No teams discovered. Run `clawteam team spawn-team` to create one.")
              : teams.map(function (t, i) {
                  const name = (t && (t.name || t.team_name)) || String(t);
                  return h(TeamRow, {
                    key: name + ":" + i,
                    team: t,
                    selected: name === selected,
                    onSelect: setSelected,
                  });
                })
          )
        ),
        h(
          Card,
          { className: "ct-detail-card" },
          h(
            CardContent,
            null,
            selected
              ? h(TeamDetail, { name: selected })
              : h("div", { className: "ct-muted" }, "Select a team to view status.")
          )
        )
      )
    );
  }

  REG.register("clawteam", App);
})();
