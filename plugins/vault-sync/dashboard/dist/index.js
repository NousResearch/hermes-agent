(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { useEffect, useState } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;

  function ArtefactCard(props) {
    const item = props.item || {};
    return h(Card, null,
      h(CardHeader, null,
        h("div", { className: "flex items-start justify-between gap-2" },
          h(CardTitle, null, props.title),
          h(Badge, { variant: item.exists ? "default" : "secondary" }, item.exists ? "present" : "missing")
        )
      ),
      h(CardContent, { className: "space-y-2 text-sm" },
        h("div", { className: "font-mono break-all text-xs text-muted-foreground" }, item.path || "—"),
        item.exists && h("div", null, `size ${item.size} · mode ${item.mode}`),
        item.sha256 && h("div", { className: "font-mono break-all text-xs" }, `sha256 ${item.sha256}`),
        h("div", { className: "text-xs text-muted-foreground" }, item.allowed ? "within allowlist" : "outside allowlist")
      )
    );
  }

  function VaultSyncPage() {
    const [state, setState] = useState({ loading: true, error: null, artefacts: null, status: null, dirs: null });
    function load() {
      setState((s) => Object.assign({}, s, { loading: true, error: null }));
      Promise.all([
        fetchJSON("/api/plugins/vault-sync/status"),
        fetchJSON("/api/plugins/vault-sync/artefacts"),
        fetchJSON("/api/plugins/vault-sync/expected-dirs"),
      ]).then(([status, artefacts, dirs]) => {
        setState({ loading: false, error: null, status, artefacts, dirs });
      }).catch((err) => {
        setState({ loading: false, error: String(err && err.message || err), status: null, artefacts: null, dirs: null });
      });
    }
    useEffect(load, []);
    const dirs = (state.dirs && state.dirs.directories) || [];
    return h("div", { className: "space-y-6" },
      h("div", { className: "flex items-start justify-between gap-4" },
        h("div", null,
          h("h1", { className: "text-2xl font-semibold" }, "Vault Sync"),
          h("p", { className: "text-sm text-muted-foreground" }, "Read-only vault artefact health. No sync, move, delete, copy, or file-content display.")
        ),
        h(Button, { onClick: load, disabled: state.loading }, state.loading ? "Refreshing…" : "Refresh")
      ),
      state.error && h(Card, { className: "border-destructive" }, h(CardContent, { className: "pt-6 text-sm text-destructive" }, state.error)),
      state.status && h("div", { className: "flex flex-wrap gap-2" },
        h(Badge, { variant: state.status.vault_root_exists ? "default" : "destructive" }, `Vault root: ${state.status.vault_root_exists ? "present" : "missing"}`),
        h(Badge, { variant: state.status.tmp_root_exists ? "default" : "secondary" }, `Temp root: ${state.status.tmp_root_exists ? "present" : "missing"}`),
        h(Badge, { variant: "outline" }, state.status.scope)
      ),
      state.artefacts && h("div", { className: "flex flex-wrap gap-2" },
        h(Badge, { variant: state.artefacts.temporary_copy_duplicates_canonical ? "destructive" : "default" }, `Temp duplicate: ${state.artefacts.temporary_copy_duplicates_canonical ? "yes" : "no"}`),
        h(Badge, { variant: "secondary" }, state.artefacts.policy)
      ),
      state.artefacts && h("div", { className: "grid gap-4 md:grid-cols-2" },
        h(ArtefactCard, { title: "Canonical SOP", item: state.artefacts.canonical }),
        h(ArtefactCard, { title: "Temporary copy", item: state.artefacts.temporary_copy })
      ),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Expected directories")),
        h(CardContent, { className: "grid gap-2 md:grid-cols-2" }, dirs.map((dir) =>
          h("div", { key: dir.name, className: "rounded-md border p-3 text-sm" },
            h("div", { className: "flex items-center justify-between gap-2" },
              h("span", { className: "font-medium" }, dir.name),
              h(Badge, { variant: dir.exists ? "default" : "secondary" }, dir.exists ? "present" : "missing")
            ),
            h("div", { className: "mt-1 font-mono text-xs text-muted-foreground break-all" }, dir.path)
          )
        ))
      ),
      state.loading && h("p", { className: "text-sm text-muted-foreground" }, "Loading vault artefact health…")
    );
  }

  Registry.register("vault-sync", VaultSyncPage);
})();
