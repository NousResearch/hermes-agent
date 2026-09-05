(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) {
    return;
  }

  const React = SDK.React;
  const { useCallback, useEffect, useState } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;
  const h = React.createElement;
  const ENDPOINT = "/api/plugins/agy-usage/usage";

  function formatDate(value) {
    if (!value) return "Unknown";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function statusLabel(status, partial) {
    if (status === "connected") return "Connected";
    if (partial || status === "unknown") return "Partial data";
    return "Unavailable";
  }

  function statusClass(status, partial) {
    if (status === "connected") return "text-emerald-600";
    if (partial || status === "unknown") return "text-amber-600";
    return "text-muted-foreground";
  }

  function QuotaWindow({ window }) {
    const used = typeof window.used_percent === "number" && Number.isFinite(window.used_percent)
      ? Math.max(0, Math.min(100, window.used_percent))
      : null;
    const remaining = used == null ? null : Math.max(0, Math.round(100 - used));
    const fillClass = remaining != null && remaining <= 20
      ? "bg-destructive"
      : remaining != null && remaining <= 50
        ? "bg-amber-500"
        : "bg-emerald-500";

    return h("div", { className: "space-y-1.5", "data-slot": "agy-quota-window" },
      h("div", { className: "flex flex-wrap justify-between gap-2 text-sm" },
        h("span", null, window.label),
        h("span", { className: "font-mono text-muted-foreground" },
          remaining == null ? "Remaining unknown" : remaining + "% remaining",
        ),
      ),
      remaining != null && h("div", {
        className: "h-2 overflow-hidden rounded-full bg-muted",
        role: "progressbar",
        "aria-label": window.label + " quota remaining",
        "aria-valuemin": 0,
        "aria-valuemax": 100,
        "aria-valuenow": remaining,
      }, h("div", {
        className: "h-full " + fillClass,
        style: { width: remaining + "%" },
      })),
      h("div", { className: "flex flex-wrap justify-between gap-2 text-xs text-muted-foreground" },
        h("span", null, used == null ? "Usage unknown" : Math.round(used) + "% used"),
        h("span", null, window.reset_at ? "Resets " + formatDate(window.reset_at) : (window.detail || "Reset unknown")),
      ),
    );
  }

  function ModelsList({ account }) {
    if (account.models && account.models.length > 0) {
      return h("div", { className: "space-y-2 border-t border-border pt-4" },
        h("div", { className: "flex items-center justify-between gap-2" },
          h("h3", { className: "text-sm font-medium" }, "Available models"),
          h("span", { className: "font-mono text-xs text-muted-foreground" }, String(account.models.length)),
        ),
        h("ul", { className: "agy-usage-models", "data-slot": "agy-model-list" },
          account.models.map((model) => h("li", { key: model.id, className: "agy-usage-model" },
            h("code", { className: "font-mono text-xs" }, model.id),
            h("span", { className: "text-xs text-muted-foreground" }, model.label),
          )),
        ),
      );
    }
    return h("p", { className: "border-t border-border pt-4 text-sm text-muted-foreground" },
      account.models_unavailable_reason || "Model inventory is unavailable.",
    );
  }

  function AccountCard({ account }) {
    const quota = account.quota || {};
    const windows = Array.isArray(quota.windows) ? quota.windows : [];
    const accountStatus = statusLabel(account.status, account.partial);
    return h(Card, { "data-slot": "agy-account-card", "data-account-id": account.id || "agy-keyring-default" },
      h(CardHeader, null,
        h("div", { className: "flex items-start justify-between gap-4" },
          h("div", { className: "min-w-0" },
            h(CardTitle, { className: "text-base" }, account.email || account.display_name || "Current agy account"),
            h("p", { className: "mt-1 font-mono text-xs text-muted-foreground" }, account.auth_source || "unknown auth source"),
          ),
          h("span", { className: "shrink-0 text-xs " + statusClass(account.status, account.partial) }, accountStatus),
        ),
      ),
      h(CardContent, { className: "space-y-4" },
        account.partial && h(Badge, { variant: "outline" }, "Some data unavailable"),
        windows.length > 0
          ? h("div", { className: "space-y-4", "data-slot": "agy-quota-windows" }, windows.map((window) => h(QuotaWindow, { key: window.label, window })))
          : h("p", { className: "text-sm text-muted-foreground" }, quota.unavailable_reason || "Quota is unavailable."),
        h(ModelsList, { account }),
        account.unavailable_reason && h("p", { className: "text-sm text-muted-foreground" }, account.unavailable_reason),
        h("p", { className: "border-t border-border pt-3 text-xs text-muted-foreground" },
          "Quota source: " + (quota.source || "Unknown") + " · Model source: " + (account.model_source || "Unknown") +
          " · Updated " + formatDate(account.fetched_at) +
          (account.scope ? " · Scope: " + account.scope : ""),
        ),
      ),
    );
  }

  function AgySummaryWindow({ window: quotaWindow }) {
    const used = typeof quotaWindow.used_percent === "number" && Number.isFinite(quotaWindow.used_percent)
      ? Math.max(0, Math.min(100, quotaWindow.used_percent))
      : null;
    const remaining = used == null ? null : Math.max(0, Math.round(100 - used));
    const fillClass = remaining != null && remaining <= 20
      ? "bg-destructive"
      : remaining != null && remaining <= 50
        ? "bg-amber-500"
        : "bg-emerald-500";
    return h("div", { className: "space-y-1.5", "data-slot": "agy-summary-window" },
      h("div", { className: "flex flex-wrap justify-between gap-2 text-sm" },
        h("span", null, quotaWindow.label),
        h("span", { className: "font-mono text-muted-foreground" },
          remaining == null ? "Remaining unknown" : remaining + "% remaining",
        ),
      ),
      remaining != null && h("div", { className: "h-2 overflow-hidden rounded-full bg-muted", role: "progressbar", "aria-label": quotaWindow.label + " quota remaining", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": remaining },
        h("div", { className: "h-full " + fillClass, style: { width: remaining + "%" } }),
      ),
      h("div", { className: "flex flex-wrap justify-between gap-2 text-xs text-muted-foreground" },
        h("span", null, used == null ? "Usage unknown" : Math.round(used) + "% used"),
        h("span", null, quotaWindow.reset_at ? "Resets " + formatDate(quotaWindow.reset_at) : (quotaWindow.detail || "Reset unknown")),
      ),
    );
  }

  function AgyUsageSummary() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const load = useCallback(function () {
      setLoading(true);
      setError(null);
      SDK.fetchJSON(ENDPOINT)
        .then(function (value) { setData(value); })
        .catch(function () { setError("AGY status is unavailable."); })
        .finally(function () { setLoading(false); });
    }, []);
    useEffect(function () { load(); }, [load]);

    const account = data && Array.isArray(data.accounts) ? data.accounts[0] : null;
    if (loading && !data) {
      return h(Card, { "data-slot": "agy-summary-card", "aria-busy": "true" },
        h(CardContent, { className: "py-8 text-center text-sm text-muted-foreground" }, "Loading Antigravity / agy…"),
      );
    }
    if (error || !account) {
      return h(Card, { "data-slot": "agy-summary-card" },
        h(CardHeader, null,
          h(CardTitle, { className: "text-base" }, "Antigravity / agy"),
          h("p", { className: "font-mono text-xs text-muted-foreground" }, "Local CLI diagnostics"),
        ),
        h(CardContent, { className: "space-y-3" },
          h("p", { className: "text-sm text-muted-foreground" }, error || "AGY status is unavailable."),
          h("a", { href: "/agy-usage", className: "text-sm font-medium text-primary underline-offset-4 hover:underline" }, "View details"),
        ),
      );
    }

    const quota = account.quota || {};
    const windows = Array.isArray(quota.windows) ? quota.windows : [];
    const status = account.status === "connected" ? "Connected" : account.partial || account.status === "unknown" ? "Partial data" : "Unavailable";
    const statusClass = account.status === "connected" ? "text-emerald-600" : account.partial || account.status === "unknown" ? "text-amber-600" : "text-muted-foreground";
    return h(Card, { "data-slot": "agy-summary-card" },
      h(CardHeader, null,
        h("div", { className: "flex items-start justify-between gap-4" },
          h("div", { className: "min-w-0" },
            h(CardTitle, { className: "text-base" }, "Antigravity / agy"),
            h("p", { className: "mt-1 font-mono text-xs text-muted-foreground" }, account.email || account.display_name || "Current agy account"),
          ),
          h("span", { className: "shrink-0 text-xs " + statusClass }, status),
        ),
      ),
      h(CardContent, { className: "space-y-4" },
        h("p", { className: "text-xs text-muted-foreground" }, "Local CLI / provider diagnostics · not Hermes session analytics"),
        windows.length > 0
          ? h("div", { className: "space-y-4", "data-slot": "agy-summary-windows" }, windows.map(function (quotaWindow) { return h(AgySummaryWindow, { key: quotaWindow.label, window: quotaWindow }); }))
          : h("p", { className: "text-sm text-muted-foreground" }, quota.unavailable_reason || "Quota is unavailable."),
        h("div", { className: "flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3 text-xs text-muted-foreground" },
          h("span", null, "Models: " + (Array.isArray(account.models) ? account.models.length : "Unknown") + " · Source: " + (quota.source || "agy /usage") + " · Updated " + formatDate(account.fetched_at)),
          h("a", { href: "/agy-usage", className: "text-sm font-medium text-primary underline-offset-4 hover:underline" }, "View details"),
        ),
      ),
    );
  }

  function AgyUsagePage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const load = useCallback(function () {
      setLoading(true);
      setError(null);
      SDK.fetchJSON(ENDPOINT)
        .then(function (value) { setData(value); })
        .catch(function (err) { setError(String(err)); })
        .finally(function () { setLoading(false); });
    }, []);

    useEffect(function () {
      load();
    }, [load]);

    const accounts = data && Array.isArray(data.accounts) ? data.accounts : [];
    return h("div", { className: "flex flex-col gap-6", "data-slot": "agy-usage-page" },
      h("div", { className: "flex flex-wrap items-start justify-between gap-3" },
        h("div", { className: "space-y-1" },
          h("h1", { className: "text-lg font-semibold" }, "Antigravity / agy"),
          h("p", { className: "text-sm text-muted-foreground" },
            "Read-only diagnostics from the local agy CLI. No model generation request is sent.",
          ),
        ),
        h(Button, { type: "button", ghost: true, onClick: load, disabled: loading, "aria-label": "Refresh agy usage" }, loading ? "Loading…" : "Refresh"),
      ),
      loading && !data && h(Card, { "aria-busy": "true" }, h(CardContent, { className: "py-12 text-center text-sm text-muted-foreground" }, "Loading agy status…")),
      error && h(Card, { "data-slot": "agy-error" }, h(CardContent, { className: "py-6" }, h("p", { className: "text-sm text-destructive" }, error))),
      data && accounts.length === 0 && h(Card, null, h(CardContent, { className: "py-12 text-center text-sm text-muted-foreground" }, "No agy account snapshot was returned.")),
      accounts.length > 0 && h("div", { className: "grid gap-6 lg:grid-cols-2", "data-slot": "agy-account-list" }, accounts.map((account) => h(AccountCard, { key: account.id || account.email || "agy-keyring-default", account }))),
    );
  }

  window.__HERMES_PLUGINS__.register("agy-usage", AgyUsagePage);
  window.__HERMES_PLUGINS__.registerSlot("agy-usage", "usage-quota:providers", AgyUsageSummary);
})();
