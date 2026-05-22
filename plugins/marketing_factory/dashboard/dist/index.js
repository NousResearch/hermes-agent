(() => {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const registry = window.__HERMES_PLUGINS__;
  if (!sdk || !registry) return;

  const React = sdk.React;
  const { useCallback, useEffect, useMemo, useState } = sdk.hooks;
  const { fetchJSON } = sdk;
  const h = React.createElement;
  const API = "/api/plugins/marketing_factory";

  // Channel character limits for displaying body-length vs cap
  const CHANNEL_LIMITS = {
    x: 280,
    instagram: 2200,
    tiktok: 2200,
    linkedin: 5000,
    blog: 5000,
    email: 5000,
    app_store: 170,
  };

  // Filter chip ordering
  const STATUS_FILTERS = ["all", "needs_review", "approved", "scheduled", "dry_run_posted", "rejected"];

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
      llm_ok: "border-emerald-300/30 bg-emerald-300/5 text-emerald-200/90",
      llm_fallback: "border-amber-300/30 bg-amber-300/5 text-amber-200/90",
      safety_ok: "border-emerald-300/20 bg-transparent text-emerald-300/90",
      safety_fail: "border-red-300/30 bg-red-300/10 text-red-100",
      route_cheap: "border-midground/20 text-midground/80",
      route_mid: "border-blue-300/20 text-blue-200/90",
      route_premium: "border-purple-300/30 text-purple-200/90",
      tone_neutral: "border-midground/20 text-midground/80",
    };
    return h("span", { className: cx("inline-flex rounded-full border px-2.5 py-1 text-xs", styles[tone] || styles.tone_neutral) }, text);
  }

  function Stat({ label, value, sublabel }) {
    return card("min-h-[92px]",
      h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60" }, label),
      h("div", { className: "mt-2 text-3xl font-semibold text-foreground" }, String(value ?? 0)),
      sublabel ? h("div", { className: "mt-1 text-xs text-midground/60" }, sublabel) : null
    );
  }

  function safetyDetail(safety) {
    if (!safety || typeof safety !== "object") return null;
    if (safety.passed) return null;
    const checks = safety.checks || {};
    const issues = [];
    if (Array.isArray(checks.forbidden_claims) && checks.forbidden_claims.length) issues.push(`forbidden: ${checks.forbidden_claims.join(", ")}`);
    if (checks.channel_constraints === false) issues.push("over channel length");
    if (checks.useful === false) issues.push("too short");
    if (checks.hallucinated_claims_risk && checks.hallucinated_claims_risk !== "low") issues.push(`hallucination risk: ${checks.hallucinated_claims_risk}`);
    return issues.length ? issues.join(" · ") : "safety check failed";
  }

  function bodyLengthIndicator(channel, body) {
    const limit = CHANNEL_LIMITS[channel] || 2000;
    const length = (body || "").length;
    const pct = Math.min(100, Math.round((length / limit) * 100));
    const tone = pct >= 100 ? "border-red-400/40 text-red-200" : pct >= 85 ? "border-amber-300/40 text-amber-200" : "border-midground/20 text-midground/70";
    return h("span", { className: cx("inline-flex rounded-full border px-2 py-0.5 text-[10px]", tone) }, `${length} / ${limit}`);
  }

  function tokenPanel(budgets) {
    const total = budgets?.spent_tokens_today || 0;
    const daily = budgets?.daily_tokens || 250000;
    const byRoute = budgets?.spent_by_route || {};
    const perApp = budgets?.per_app_tokens || {};
    const perChannel = budgets?.per_channel_tokens || {};
    const pct = Math.min(100, Math.round((total / daily) * 100));
    return card(null,
      h("div", { className: "flex items-center justify-between gap-2" },
        h("h2", { className: "text-lg font-semibold" }, "Token spend"),
        h("span", { className: "text-xs text-midground/60" }, `daily limit ${daily.toLocaleString()}`)
      ),
      h("div", { className: "mt-3 flex items-baseline gap-2" },
        h("div", { className: "text-3xl font-semibold text-foreground" }, total.toLocaleString()),
        h("div", { className: "text-xs text-midground/70" }, `tokens today (${pct}% of cap)`)
      ),
      h("div", { className: "mt-3 h-2 w-full overflow-hidden rounded-full bg-midground/10" },
        h("div", {
          className: cx("h-full", pct >= 90 ? "bg-red-400/70" : pct >= 70 ? "bg-amber-300/70" : "bg-cyan-300/60"),
          style: { width: `${pct}%` },
        })
      ),
      h("div", { className: "mt-3 grid grid-cols-3 gap-2 text-xs" },
        ["cheap", "mid", "premium"].map((route) => h("div", {
          key: route,
          className: "rounded-lg border border-midground/15 bg-background/60 p-2",
        },
          h("div", { className: "uppercase tracking-[0.16em] text-midground/60" }, route),
          h("div", { className: "mt-1 text-sm font-semibold" }, (byRoute[route] || 0).toLocaleString())
        ))
      ),
      Object.keys(perApp).length ? h("div", { className: "mt-3 text-xs text-midground/70" },
        h("div", { className: "uppercase tracking-[0.14em] text-midground/50" }, "by app"),
        h("div", { className: "mt-1 flex flex-wrap gap-2" },
          Object.entries(perApp).map(([slug, n]) => h("span", {
            key: slug,
            className: "rounded-full border border-midground/15 px-2 py-0.5",
          }, `${slug}: ${n.toLocaleString()}`))
        )
      ) : null,
      Object.keys(perChannel).length ? h("div", { className: "mt-2 text-xs text-midground/70" },
        h("div", { className: "uppercase tracking-[0.14em] text-midground/50" }, "by channel"),
        h("div", { className: "mt-1 flex flex-wrap gap-2" },
          Object.entries(perChannel).map(([ch, n]) => h("span", {
            key: ch,
            className: "rounded-full border border-midground/15 px-2 py-0.5",
          }, `${ch === "_none" ? "(research)" : ch}: ${n.toLocaleString()}`))
        )
      ) : null,
      budgets?.last_reset_date ? h("div", { className: "mt-3 text-[10px] text-midground/50" }, `daily counters last reset ${budgets.last_reset_date} (UTC)`) : null
    );
  }

  function channelModePanel(app, busy, run) {
    if (!app) return null;
    const modes = app.channel_modes || {};
    const channels = app.channels || [];
    if (!channels.length) return null;
    const anyLive = channels.some((channel) => (modes[channel] || "dry_run") === "live");
    return card(anyLive ? "border-red-400/40 bg-red-500/5" : null,
      h("div", { className: "flex items-center justify-between gap-2" },
        h("h2", { className: "text-lg font-semibold" }, "Channel publish modes"),
        anyLive ? h("span", { className: "text-xs text-red-200 font-medium" }, "LIVE channel(s) active") : h("span", { className: "text-xs text-midground/60" }, "All channels dry-run")
      ),
      h("p", { className: "mt-2 text-xs text-midground/70" }, "Until a real connector is registered, switching a channel to live still falls back to dry-run automatically (audited). When the connector IS wired, live = real public posts."),
      h("div", { className: "mt-3 flex flex-col gap-2" },
        channels.map((channel) => {
          const mode = modes[channel] || "dry_run";
          const live = mode === "live";
          return h("div", {
            key: channel,
            className: cx("flex items-center justify-between rounded-xl border p-2.5", live ? "border-red-400/40 bg-red-500/10" : "border-midground/15 bg-background/60"),
          },
            h("div", { className: "flex items-center gap-2" },
              h("span", { className: "text-sm font-medium" }, channel),
              h("span", { className: cx("inline-flex rounded-full border px-2 py-0.5 text-[10px]", live ? "border-red-300/40 bg-red-300/10 text-red-100" : "border-midground/20 text-midground/70") }, mode)
            ),
            h("button", {
              type: "button",
              disabled: !!busy,
              onClick: () => {
                const nextMode = live ? "dry_run" : "live";
                if (nextMode === "live") {
                  const confirmText = window.prompt(`Switch ${channel} to LIVE for ${app.slug}? Live means real public posts when a connector is registered.\n\nType the channel name (${channel}) to confirm:`);
                  if (!confirmText || confirmText.trim() !== channel) return;
                }
                return run(`mode ${channel}`, () => fetchJSON(`${API}/apps/${encodeURIComponent(app.slug)}/channels/${encodeURIComponent(channel)}/mode`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: nextMode, reviewer: "dashboard" }) }));
              },
              className: cx(
                "min-h-[32px] rounded-lg border px-2.5 py-1 text-xs font-medium",
                live ? "border-red-400/40 text-red-100 hover:bg-red-500/10" : "border-midground/20 text-foreground hover:bg-midground/10"
              ),
            }, live ? "switch to dry-run" : "switch to LIVE")
          );
        })
      )
    );
  }

  function MarketingFactoryPage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState(null);
    const [error, setError] = useState(null);
    const [appSlug, setAppSlug] = useState("pupular");
    const [days, setDays] = useState(7);
    const [statusFilter, setStatusFilter] = useState("needs_review");

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

    const allDrafts = useMemo(() => {
      const drafts = data?.drafts || [];
      return drafts.filter((draft) => !appSlug || draft.app_slug === appSlug).slice().reverse();
    }, [data, appSlug]);

    const filteredDrafts = useMemo(() => {
      if (statusFilter === "all") return allDrafts;
      return allDrafts.filter((draft) => draft.status === statusFilter);
    }, [allDrafts, statusFilter]);

    const statusCounts = useMemo(() => {
      const counts = { all: allDrafts.length };
      for (const draft of allDrafts) {
        counts[draft.status] = (counts[draft.status] || 0) + 1;
      }
      return counts;
    }, [allDrafts]);

    const pending = allDrafts.filter((draft) => draft.status === "needs_review");
    const approved = allDrafts.filter((draft) => draft.status === "approved");
    const scheduled = allDrafts.filter((draft) => draft.status === "scheduled");

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

      (() => {
        const advisor = data?.advisor;
        if (!advisor || advisor.healthy) return null;
        const items = advisor.items || [];
        const warnings = items.filter((item) => item.severity === "warning");
        const headerTone = warnings.length ? "border-amber-300/40 bg-amber-300/10 text-amber-100" : "border-blue-300/30 bg-blue-300/5 text-blue-100";
        return h("div", { className: cx("rounded-xl border p-3 text-sm", headerTone) },
          h("div", { className: "flex flex-wrap items-baseline gap-2" },
            h("span", { className: "font-semibold" }, `${items.length} advisor item${items.length === 1 ? "" : "s"}`),
            warnings.length ? h("span", { className: "text-xs uppercase tracking-[0.16em]" }, `${warnings.length} warning${warnings.length === 1 ? "" : "s"}`) : null
          ),
          h("ul", { className: "mt-2 flex flex-col gap-2 text-xs" },
            items.map((item, idx) => h("li", { key: idx, className: "rounded-lg border border-midground/15 bg-background/40 p-2" },
              h("div", { className: "flex flex-wrap items-baseline gap-2" },
                h("span", { className: cx("inline-flex rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.14em]", item.severity === "warning" ? "border-amber-300/40 text-amber-200" : "border-blue-300/30 text-blue-200") }, item.severity),
                item.app_slug ? h("span", { className: "text-midground/70" }, item.app_slug) : null,
                h("span", null, item.message)
              ),
              h("div", { className: "mt-1 text-midground/70" }, `→ ${item.action}`)
            ))
          )
        );
      })(),

      h("div", { className: "grid gap-3 sm:grid-cols-2 lg:grid-cols-5" },
        h(Stat, { label: "Apps", value: data?.summary?.apps }),
        h(Stat, { label: "Campaigns", value: data?.summary?.campaigns }),
        h(Stat, { label: "Drafts", value: data?.summary?.drafts }),
        h(Stat, { label: "Pending", value: data?.summary?.pending_approvals }),
        h(Stat, { label: "Dry-runs", value: data?.summary?.dry_run_publish_events })
      ),

      (() => {
        const poll = data?.summary?.poll || {};
        const lastPolledAt = poll.last_poll_at ? new Date(poll.last_poll_at).toLocaleString() : "never";
        const lastFired = poll.last_poll_fired ?? 0;
        const totalPolls = poll.total_polls ?? 0;
        return card("border-midground/15",
          h("div", { className: "flex flex-wrap items-center justify-between gap-2" },
            h("div", null,
              h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60" }, "Scheduled poller"),
              h("div", { className: "mt-1 text-sm text-midground" }, `Last tick: ${lastPolledAt} · fired ${lastFired} on last tick · ${totalPolls} ticks total`)
            ),
            h("div", { className: "flex flex-wrap gap-2" },
              smallButton("Run poll now", () => run("poll", () => fetchJSON(`${API}/poll`, { method: "POST" })), !!busy),
              h("code", { className: "rounded-lg border border-midground/15 bg-background/60 px-2 py-1 text-[10px] text-midground/70" }, 'hermes cron create --schedule "every 5m" --command "hermes marketing-factory poll"')
            )
          )
        );
      })(),

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
            smallButton(`Dry-run publish (${scheduled.length})`, () => run("publish", () => fetchJSON(`${API}/publish-dry-run?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" })), !!busy || !scheduled.length),
            smallButton(`Publish — mode-aware (${scheduled.length})`, () => {
              const currentApp = (data?.apps || []).find((a) => a.slug === appSlug);
              const liveChannels = currentApp ? (currentApp.channels || []).filter((c) => (currentApp.channel_modes || {})[c] === "live") : [];
              if (liveChannels.length) {
                const ok = window.confirm(`LIVE channels active for ${appSlug}: ${liveChannels.join(", ")}.\n\nIf a real connector is registered, this WILL post publicly. Without a connector it falls back to dry-run (audited). Proceed?`);
                if (!ok) return;
              }
              return run("publish-modes", () => fetchJSON(`${API}/publish?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" }));
            }, !!busy || !scheduled.length)
          )
        )
      ),

      h("div", { className: "grid gap-4 xl:grid-cols-[1.45fr_0.85fr]" },
        card(null,
          h("div", { className: "flex flex-wrap items-center justify-between gap-3" },
            h("h2", { className: "text-lg font-semibold" }, `Draft queue${appSlug ? ` · ${appSlug}` : ""}`),
            h("div", { className: "flex flex-wrap items-center gap-2" },
              smallButton(`Approve all pending (${pending.length})`, () => run("approve-all", () => fetchJSON(`${API}/drafts/approve-all?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" })), !!busy || !pending.length, "primary"),
              smallButton(`Reject all pending (${pending.length})`, () => run("reject-all", () => fetchJSON(`${API}/drafts/reject-all?app_slug=${encodeURIComponent(appSlug)}`, { method: "POST" })), !!busy || !pending.length, "danger")
            )
          ),
          h("div", { className: "mt-3 flex flex-wrap gap-2" },
            STATUS_FILTERS.map((status) => h("button", {
              key: status,
              type: "button",
              onClick: () => setStatusFilter(status),
              className: cx(
                "min-h-[36px] rounded-full border px-3 py-1 text-xs transition",
                statusFilter === status ? "border-cyan-300/40 bg-cyan-300/10 text-cyan-100" : "border-midground/20 text-midground hover:bg-midground/10"
              ),
            }, `${status === "all" ? "All" : status.replace(/_/g, " ")} · ${statusCounts[status] || 0}`))
          ),
          h("div", { className: "mt-4 flex flex-col gap-3" },
            filteredDrafts.length ? filteredDrafts.slice(0, 30).map((draft) => h("article", { key: draft.id, className: "rounded-xl border border-midground/15 bg-background/70 p-3" },
              h("div", { className: "flex flex-wrap items-center gap-2" },
                pill(draft.status, draft.status),
                h("span", { className: "text-xs uppercase tracking-[0.14em] text-midground/60" }, draft.channel),
                pill(draft.model_route || "cheap", `route_${draft.model_route || "cheap"}`),
                draft.llm_used ? pill(`llm: ${draft.llm_model || "ok"}`, "llm_ok") : pill(draft.llm_error ? "template (llm err)" : "template", draft.llm_error ? "llm_fallback" : "tone_neutral"),
                draft.safety ? pill(draft.safety.passed ? "safety ok" : "safety fail", draft.safety.passed ? "safety_ok" : "safety_fail") : null,
                bodyLengthIndicator(draft.channel, draft.body),
                h("span", { className: "text-xs text-midground/60" }, draft.scheduled_for ? new Date(draft.scheduled_for).toLocaleString() : "unscheduled")
              ),
              draft.llm_error ? h("div", { className: "mt-2 text-[11px] text-amber-200/90" }, `LLM fallback reason: ${draft.llm_error}`) : null,
              draft.safety && !draft.safety.passed && safetyDetail(draft.safety) ? h("div", { className: "mt-2 text-[11px] text-red-200/90" }, `Safety issues: ${safetyDetail(draft.safety)}`) : null,
              h("p", { className: "mt-3 whitespace-pre-wrap text-sm leading-6" }, draft.body),
              h("div", { className: "mt-3 flex flex-wrap gap-2" },
                smallButton("Approve", () => run(`approve ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: "Approved in Marketing Factory dashboard" }) })), !!busy || draft.status !== "needs_review", "primary"),
                smallButton("Reject", () => {
                  const reason = window.prompt("Why are you rejecting this draft? Specific feedback steers future generations.\n\n(Leave blank to cancel.)");
                  if (!reason || !reason.trim()) return;
                  return run(`reject ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/reject`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: reason.trim() }) }));
                }, !!busy || draft.status !== "needs_review", "danger"),
                smallButton("Regenerate", () => run(`regen ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/regenerate`, { method: "POST" })), !!busy || draft.status === "scheduled" || draft.status === "dry_run_posted" || draft.status === "posted"),
                smallButton("Schedule", () => run(`schedule ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/schedule`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) })), !!busy || draft.status !== "approved"),
                smallButton("Dry-run", () => run(`dry-run ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/publish-dry-run`, { method: "POST" })), !!busy || draft.status !== "scheduled")
              )
            )) : h("p", { className: "text-sm text-midground" }, statusFilter === "all" ? "No drafts yet. Initialize samples, then generate a campaign." : `No drafts with status “${statusFilter.replace(/_/g, " ")}”.`)
          )
        ),

        h("div", { className: "flex flex-col gap-4" },
          tokenPanel(data?.summary?.budgets),
          channelModePanel((data?.apps || []).find((app) => app.slug === appSlug), busy, run),
          card(null,
            h("div", { className: "flex items-center justify-between gap-2" },
              h("h2", { className: "text-lg font-semibold" }, "Brand apps"),
              smallButton("+ Add app", () => {
                const slug = window.prompt("New app slug (lowercase, alphanumeric, dashes/underscores ok):");
                if (!slug || !slug.trim()) return;
                const name = window.prompt(`Display name for ${slug}:`);
                if (!name || !name.trim()) return;
                const positioning = window.prompt("Brand positioning (one sentence):") || "";
                const icp = window.prompt("Ideal customer profile (who buys/uses this?):") || "";
                const tone = window.prompt("Brand tone (e.g. cute warm playful / trustworthy premium clear):") || "";
                const cta = window.prompt("Default call-to-action:") || "";
                const channelsStr = window.prompt("Channels (comma-separated, e.g. x,instagram,tiktok,blog,email,linkedin,app_store):") || "";
                const channels = channelsStr.split(",").map((s) => s.trim()).filter(Boolean);
                if (!channels.length) {
                  alert("At least one channel is required. Aborting.");
                  return;
                }
                const pillarsStr = window.prompt("Content pillars (comma-separated, the recurring themes):") || "";
                const forbiddenStr = window.prompt("Forbidden claims (comma-separated, things the brand must NEVER promise):") || "";
                const linkStr = window.prompt("Primary link (e.g. App Store URL, website):") || "";
                return run("add-app", () => fetchJSON(`${API}/apps`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
                  slug: slug.trim(), name: name.trim(), positioning, icp, tone, cta,
                  channels,
                  content_pillars: pillarsStr.split(",").map((s) => s.trim()).filter(Boolean),
                  forbidden_claims: forbiddenStr.split(",").map((s) => s.trim()).filter(Boolean),
                  links: linkStr ? [linkStr] : [],
                  claims: [], competitors: [], assets: [],
                }) }));
              }, !!busy, "primary")
            ),
            h("div", { className: "mt-3 flex flex-col gap-3" },
              (data?.apps || []).map((app) => h("div", { key: app.slug, className: cx("rounded-xl border p-3 text-sm", app.slug === appSlug ? "border-cyan-300/40 bg-cyan-300/10" : "border-midground/15 bg-background/60") },
                h("div", { className: "flex items-start justify-between gap-2" },
                  h("button", { type: "button", onClick: () => setAppSlug(app.slug), className: "flex-1 text-left" },
                    h("div", { className: "font-semibold" }, app.name || app.slug),
                    h("div", { className: "mt-1 text-midground" }, app.positioning || app.icp || "No positioning set")
                  ),
                  h("div", { className: "flex flex-col gap-1 items-end" },
                    h("button", {
                      type: "button",
                      disabled: !!busy,
                      onClick: async () => {
                        try {
                          const result = await fetchJSON(`${API}/apps/${encodeURIComponent(app.slug)}/digest?days=7`);
                          if (navigator.clipboard && navigator.clipboard.writeText) {
                            await navigator.clipboard.writeText(result.markdown);
                            alert(`7-day digest for ${app.slug} copied to clipboard.`);
                          } else {
                            window.prompt("Copy this markdown digest:", result.markdown);
                          }
                        } catch (err) {
                          alert(`Digest failed: ${err.message || err}`);
                        }
                      },
                      className: "text-[10px] text-midground/60 hover:text-cyan-200 underline",
                    }, "copy digest"),
                    h("button", {
                      type: "button",
                      disabled: !!busy,
                      onClick: () => {
                        const confirmText = window.prompt(`Type the slug (${app.slug}) to DELETE this app and ALL its drafts/campaigns/etc:`);
                        if (!confirmText || confirmText.trim() !== app.slug) return;
                        return run(`remove ${app.slug}`, () => fetchJSON(`${API}/apps/${encodeURIComponent(app.slug)}?cascade=true`, { method: "DELETE" }));
                      },
                      className: "text-[10px] text-midground/60 hover:text-red-300 underline",
                    }, "remove")
                  )
                )
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
