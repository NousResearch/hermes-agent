(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { Button, Badge } = SDK.components;
  const { useEffect, useState, useCallback } = SDK.hooks;

  function Dashboard() {
    const [payload, setPayload] = useState(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const load = useCallback(async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJSON("/api/plugins/kb-live-dashboard/live?limit=8");
        setPayload(data && data.payload ? data.payload : data);
      } catch (err) {
        setError(err && err.message ? err.message : "Dashboard unavailable");
      } finally {
        setLoading(false);
      }
    }, []);

    useEffect(() => {
      load();
      const timer = window.setInterval(load, 60000);
      return () => window.clearInterval(timer);
    }, [load]);

    const summary = payload && payload.summary ? payload.summary : {};
    const sections = payload && Array.isArray(payload.sections) ? payload.sections : [];
    return h("div", { className: "kb-live" },
      h("div", { className: "kb-live-header" },
        h("div", null,
          h("h1", null, "KB Dashboard"),
          h("p", null, "Live attention, queue, workflow, and publication state from kb-engine.")
        ),
        h(Button, { onClick: load, disabled: loading }, loading ? "Refreshing" : "Refresh")
      ),
      error ? h("div", { className: "kb-live-error" }, error) : null,
      h("div", { className: "kb-live-metrics" },
        metric("Readiness", summary.readiness_status || "unknown"),
        metric("Publication", summary.publication_status || "unknown"),
        metric("Queue", summary.queue_item_count),
        metric("TODOs", summary.active_todo_count || summary.triage_todo_count),
        metric("Runs", summary.active_run_count)
      ),
      h("div", { className: "kb-live-sections" },
        sections.map((section) => sectionView(section))
      ),
      payload ? h("div", { className: "kb-live-footer" },
        "Generated " + (payload.generated_at || "unknown") + " · refresh target " +
        ((payload.refresh && payload.refresh.ttl_seconds) || 60) + "s"
      ) : null
    );
  }

  function metric(label, value) {
    return h("div", { className: "kb-live-metric", key: label },
      h("span", null, label),
      h("strong", null, value === undefined || value === null ? "unknown" : String(value))
    );
  }

  function sectionView(section) {
    const cards = section && Array.isArray(section.cards) ? section.cards : [];
    return h("section", { className: "kb-live-section", key: section.id || section.title },
      h("div", { className: "kb-live-section-head" },
        h("h2", null, section.title || section.id || "Section"),
        section.hidden_by_feedback ? h(Badge, { variant: "secondary" }, String(section.hidden_by_feedback) + " hidden") : null
      ),
      cards.length ? cards.map(cardView) : h("p", { className: "kb-live-empty" }, "No current items.")
    );
  }

  function cardView(card) {
    const severity = card.severity || "normal";
    return h("article", { className: "kb-live-card kb-live-card-" + severity, key: card.id || card.title },
      h("div", { className: "kb-live-card-main" },
        h("h3", null, card.title || "Item"),
        card.detail ? h("p", null, card.detail) : null
      ),
      h("div", { className: "kb-live-card-meta" },
        card.kind ? h(Badge, { variant: "secondary" }, card.kind) : null,
        card.target ? h("code", null, card.target) : null
      )
    );
  }

  window.__HERMES_PLUGINS__.register("kb-live-dashboard", Dashboard);
})();
