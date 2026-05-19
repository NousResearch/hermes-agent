(() => {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const registry = window.__HERMES_PLUGINS__;
  if (!sdk || !registry) return;

  const React = sdk.React;
  const { useCallback, useEffect, useMemo, useState } = sdk.hooks;
  const { fetchJSON } = sdk;
  const h = React.createElement;
  const API = "/api/plugins/marketing_factory";

  function cx(...parts) {
    return parts.filter(Boolean).join(" ");
  }

  function card(className, ...children) {
    return h("section", { className: cx("rounded-2xl border border-midground/15 bg-background/65 p-4 shadow-sm", className) }, children);
  }

  function smallButton(label, onClick, disabled, tone) {
    return h("button", {
      type: "button",
      onClick,
      disabled,
      className: cx(
        "min-h-[44px] rounded-xl border px-3 py-2 text-sm font-medium transition",
        disabled ? "cursor-not-allowed border-midground/10 text-midground/35" : "border-midground/20 text-foreground hover:bg-midground/10",
        tone === "danger" && !disabled ? "border-red-400/40 text-red-200 hover:bg-red-500/10" : "",
        tone === "primary" && !disabled ? "border-cyan-300/40 bg-cyan-300/10 text-cyan-100 hover:bg-cyan-300/20" : ""
      ),
    }, label);
  }

  function pill(text, tone) {
    const styles = {
      needs_review: "border-amber-300/30 bg-amber-300/10 text-amber-100",
      approved: "border-emerald-300/30 bg-emerald-300/10 text-emerald-100",
      scheduled: "border-blue-300/30 bg-blue-300/10 text-blue-100",
      dry_run_posted: "border-purple-300/30 bg-purple-300/10 text-purple-100",
      rejected: "border-red-300/30 bg-red-300/10 text-red-100",
    };
    return h("span", { className: cx("inline-flex rounded-full border px-2.5 py-1 text-xs", styles[tone] || "border-midground/20 text-midground") }, text);
  }

  function Stat({ label, value }) {
    return card("min-h-[92px]", h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60" }, label), h("div", { className: "mt-2 text-3xl font-semibold text-foreground" }, String(value ?? 0)));
  }

  function MarketingFactoryPage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState(null);
    const [error, setError] = useState(null);
    const [appSlug, setAppSlug] = useState("pupular");
    const [days, setDays] = useState(7);

    const refresh = useCallback(async () => {
      setError(null);
      const next = await fetchJSON(`${API}/overview`);
      setData(next);
      if (next.apps && next.apps.length && !next.apps.find((app) => app.slug === appSlug)) {
        setAppSlug(next.apps[0].slug);
      }
    }, [appSlug]);

    useEffect(() => {
      refresh().catch((err) => setError(err.message || String(err))).finally(() => setLoading(false));
    }, []);

    const run = useCallback(async (label, fn) => {
      setBusy(label);
      setError(null);
      try {
        const res = await fn();
        if (res && res.overview) setData(res.overview);
        else await refresh();
      } catch (err) {
        setError(err.message || String(err));
      } finally {
        setBusy(null);
      }
    }, [refresh]);

    const selectedDrafts = useMemo(() => {
      const drafts = data?.drafts || [];
      return drafts.filter((draft) => !appSlug || draft.app_slug === appSlug).slice().reverse();
    }, [data, appSlug]);

    const pending = selectedDrafts.filter((draft) => draft.status === "needs_review");
    const approved = selectedDrafts.filter((draft) => draft.status === "approved");
    const scheduled = selectedDrafts.filter((draft) => draft.status === "scheduled");

    if (loading) return h("div", { className: "p-4 text-sm text-midground" }, "Loading Marketing Factory…");

    return h("div", { className: "mx-auto flex w-full max-w-7xl flex-col gap-4 p-4 text-foreground" },
      h("header", { className: "flex flex-col gap-3 md:flex-row md:items-end md:justify-between" },
        h("div", null,
          h("div", { className: "text-xs uppercase tracking-[0.22em] text-cyan-200/70" }, "Dry-run-first Mission Control"),
          h("h1", { className: "mt-1 text-3xl font-semibold" }, "Marketing Factory"),
          h("p", { className: "mt-2 max-w-3xl text-sm leading-6 text-midground" }, "Review brand profiles, generate campaign drafts, approve safely, schedule, and dry-run publish. This MVP never performs public posting.")
        ),
        h("div", { className: "flex flex-wrap gap-2" },
          smallButton("Refresh", () => run("refresh", refresh), !!busy),
          smallButton("Initialize samples", () => run("init", () => fetchJSON(`${API}/init`, { method: "POST" })), !!busy, "primary")
        )
      ),

      error ? h("div", { className: "rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-100", role: "alert" }, error) : null,
      busy ? h("div", { className: "rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-sm text-cyan-100" }, `Working: ${busy}…`) : null,

      h("div", { className: "grid gap-3 sm:grid-cols-2 lg:grid-cols-5" },
        h(Stat, { label: "Apps", value: data?.summary?.apps }),
        h(Stat, { label: "Campaigns", value: data?.summary?.campaigns }),
        h(Stat, { label: "Drafts", value: data?.summary?.drafts }),
        h(Stat, { label: "Pending", value: data?.summary?.pending_approvals }),
        h(Stat, { label: "Dry-runs", value: data?.summary?.dry_run_publish_events })
      ),

      card("border-cyan-300/20 bg-cyan-300/5",
        h("div", { className: "flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between" },
          h("div", null,
            h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60" }, "Next marketing action"),
            h("div", { className: "mt-1 text-xl font-semibold" }, data?.next_action?.title || "Inspect state"),
            h("p", { className: "mt-1 text-sm text-midground" }, data?.next_action?.detail || "No recommendation available.")
          ),
          h("div", { className: "flex flex-wrap items-center gap-2" },
            h("select", { value: appSlug, onChange: (e) => setAppSlug(e.target.value), className: "min-h-[44px] rounded-xl border border-midground/20 bg-background px-3 text-sm" },
              (data?.apps || []).map((app) => h("option", { key: app.slug, value: app.slug }, app.name || app.slug))
            ),
            h("input", { type: "number", min: 1, max: 31, value: days, onChange: (e) => setDays(Number(e.target.value || 7)), className: "min-h-[44px] w-20 rounded-xl border border-midground/20 bg-background px-3 text-sm" }),
            smallButton("Generate", () => run("generate", () => fetchJSON(`${API}/campaigns/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ app_slug: appSlug, days }) })), !!busy || !appSlug, "primary"),
            smallButton(`Schedule approved (${approved.length})`, () => run("schedule", () => fetchJSON(`${API}/schedule?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" })), !!busy || !approved.length),
            smallButton(`Dry-run publish (${scheduled.length})`, () => run("publish", () => fetchJSON(`${API}/publish-dry-run?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" })), !!busy || !scheduled.length)
          )
        )
      ),

      h("div", { className: "grid gap-4 xl:grid-cols-[1.45fr_0.85fr]" },
        card(null,
          h("div", { className: "flex items-center justify-between gap-3" },
            h("h2", { className: "text-lg font-semibold" }, `Draft queue${appSlug ? ` · ${appSlug}` : ""}`),
            h("span", { className: "text-sm text-midground" }, `${selectedDrafts.length} shown`)
          ),
          h("div", { className: "mt-4 flex flex-col gap-3" },
            selectedDrafts.length ? selectedDrafts.slice(0, 18).map((draft) => h("article", { key: draft.id, className: "rounded-xl border border-midground/15 bg-background/70 p-3" },
              h("div", { className: "flex flex-wrap items-center gap-2" },
                pill(draft.status, draft.status),
                h("span", { className: "text-xs uppercase tracking-[0.14em] text-midground/60" }, draft.channel),
                h("span", { className: "text-xs text-midground/60" }, draft.scheduled_for || "unscheduled")
              ),
              h("p", { className: "mt-3 whitespace-pre-wrap text-sm leading-6" }, draft.body),
              h("div", { className: "mt-3 flex flex-wrap gap-2" },
                smallButton("Approve", () => run(`approve ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: "Approved in Marketing Factory dashboard" }) })), !!busy || draft.status !== "needs_review", "primary"),
                smallButton("Reject", () => run(`reject ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/reject`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: "Rejected in Marketing Factory dashboard" }) })), !!busy || draft.status !== "needs_review", "danger"),
                smallButton("Schedule", () => run(`schedule ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/schedule`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) })), !!busy || draft.status !== "approved"),
                smallButton("Dry-run", () => run(`dry-run ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/publish-dry-run`, { method: "POST" })), !!busy || draft.status !== "scheduled")
              )
            )) : h("p", { className: "text-sm text-midground" }, "No drafts yet. Initialize samples, then generate a campaign.")
          )
        ),

        h("div", { className: "flex flex-col gap-4" },
          card(null,
            h("h2", { className: "text-lg font-semibold" }, "Brand apps"),
            h("div", { className: "mt-3 flex flex-col gap-3" },
              (data?.apps || []).map((app) => h("button", { key: app.slug, onClick: () => setAppSlug(app.slug), className: cx("min-h-[44px] rounded-xl border p-3 text-left text-sm", app.slug === appSlug ? "border-cyan-300/40 bg-cyan-300/10" : "border-midground/15 bg-background/60") },
                h("div", { className: "font-semibold" }, app.name || app.slug),
                h("div", { className: "mt-1 text-midground" }, app.positioning || app.icp || "No positioning set")
              ))
            )
          ),
          card(null,
            h("h2", { className: "text-lg font-semibold" }, "Audit trail"),
            h("div", { className: "mt-3 max-h-[420px] overflow-auto rounded-xl border border-midground/10" },
              (data?.audit || []).slice().reverse().map((event) => h("div", { key: event.id, className: "border-b border-midground/10 p-3 text-sm last:border-0" },
                h("div", { className: "flex items-center justify-between gap-2" },
                  h("span", { className: "font-medium" }, event.action),
                  h("span", { className: "text-xs text-midground/60" }, event.app_slug || "system")
                ),
                h("div", { className: "mt-1 text-xs text-midground/60" }, event.timestamp)
              ))
            )
          )
        )
      )
    );
  }

  registry.register("marketing_factory", MarketingFactoryPage);
})();
