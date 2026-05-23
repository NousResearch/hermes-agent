(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { Button, Badge } = SDK.components;
  const { useEffect, useState, useCallback } = SDK.hooks;

  function DashboardBridge() {
    const [info, setInfo] = useState(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const load = useCallback(async () => {
      setLoading(true);
      setError("");
      try {
        setInfo(await fetchJSON("/api/plugins/kb-live-dashboard/standalone"));
      } catch (err) {
        setInfo(null);
        setError(err && err.message ? err.message : "Standalone dashboard link unavailable");
      } finally {
        setLoading(false);
      }
    }, []);

    useEffect(() => {
      load();
    }, [load]);

    const url = info && info.url ? info.url : "";
    return h("div", { className: "kb-live kb-live-bridge" },
      h("div", { className: "kb-live-header" },
        h("div", null,
          h("h1", null, "KB Dashboard"),
          h("p", null, "The production KB dashboard now lives as a standalone live app.")
        ),
        h(Button, { onClick: load, disabled: loading }, loading ? "Checking" : "Check Link")
      ),
      h("section", { className: "kb-live-bridge-panel" },
        h(Badge, { variant: "secondary" }, "Deprecated Hermes tab"),
        h("h2", null, "Open the standalone dashboard"),
        h("p", null, "Hermes remains the chat projection consumer. The visual/live dashboard is served by kb-dashboard on helix and reads canonical kb-engine projection packets."),
        url ? h("a", { className: "kb-live-bridge-link", href: url, target: "_blank", rel: "noreferrer" }, url) : null,
        error ? h("div", { className: "kb-live-error" }, error) : null
      )
    );
  }

  window.__HERMES_PLUGINS__.register("kb-live-dashboard", DashboardBridge);
})();
