(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { useEffect, useState } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;

  function Field(props) {
    return h("div", { className: "text-sm" },
      h("div", { className: "text-xs uppercase tracking-wide text-muted-foreground" }, props.label),
      h("div", { className: "font-mono break-all" }, props.value == null ? "—" : String(props.value))
    );
  }

  function StatBlock(props) {
    const meta = props.meta || {};
    const status = meta.exists ? "present" : "missing";
    return h("div", { className: "rounded-md border p-3 space-y-1" },
      h("div", { className: "flex items-center justify-between gap-2" },
        h("span", { className: "font-medium" }, props.label),
        h(Badge, { variant: meta.exists ? "default" : "secondary" }, status)
      ),
      h("div", { className: "text-xs text-muted-foreground" }, meta.exists ? `size ${meta.size} · mode ${meta.mode}` : "not found")
    );
  }

  function ProfileManagerPage() {
    const [state, setState] = useState({ loading: true, error: null, data: null });
    function load() {
      setState((s) => Object.assign({}, s, { loading: true, error: null }));
      fetchJSON("/api/plugins/profile-manager/inventory")
        .then((data) => setState({ loading: false, error: null, data }))
        .catch((err) => setState({ loading: false, error: String(err && err.message || err), data: null }));
    }
    useEffect(load, []);
    const profiles = (state.data && state.data.profiles) || [];
    return h("div", { className: "space-y-6" },
      h("div", { className: "flex items-start justify-between gap-4" },
        h("div", null,
          h("h1", { className: "text-2xl font-semibold" }, "Profiles (GSSAI)"),
          h("p", { className: "text-sm text-muted-foreground" }, "Read-only profile inventory. Secret files are metadata-only; no token values are displayed.")
        ),
        h(Button, { onClick: load, disabled: state.loading }, state.loading ? "Refreshing…" : "Refresh")
      ),
      state.error && h(Card, { className: "border-destructive" }, h(CardContent, { className: "pt-6 text-sm text-destructive" }, state.error)),
      state.data && h("div", { className: "flex flex-wrap gap-2" },
        h(Badge, null, state.data.scope),
        h(Badge, { variant: "secondary" }, state.data.policy),
        h(Badge, { variant: "outline" }, `${profiles.length} profiles`)
      ),
      h("div", { className: "grid gap-4 md:grid-cols-2 xl:grid-cols-3" }, profiles.map((profile) =>
        h(Card, { key: profile.name },
          h(CardHeader, null,
            h("div", { className: "flex items-start justify-between gap-2" },
              h("div", null,
                h(CardTitle, null, profile.name),
                h("p", { className: "text-xs text-muted-foreground" }, profile.boundary_label)
              ),
              h(Badge, { variant: profile.lifecycle === "active" ? "default" : "secondary" }, profile.lifecycle)
            )
          ),
          h(CardContent, { className: "space-y-3" },
            h(Field, { label: "Path", value: profile.path }),
            h("div", { className: "grid grid-cols-2 gap-2" },
              h(StatBlock, { label: ".env", meta: profile.env }),
              h(StatBlock, { label: "config.yaml", meta: profile.config_yaml }),
              h(StatBlock, { label: "SOUL.md", meta: profile.soul_md }),
              h(StatBlock, { label: "state.db", meta: profile.state_db }),
              h(StatBlock, { label: "gateway.log", meta: profile.gateway_log })
            )
          )
        )
      )),
      state.loading && h("p", { className: "text-sm text-muted-foreground" }, "Loading profile inventory…")
    );
  }

  Registry.register("profile-manager", ProfileManagerPage);
})();
