(() => {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const registry = window.__HERMES_PLUGINS__;
  if (!sdk || !registry) return;

  const React = sdk.React;
  const { useCallback, useEffect, useMemo, useState } = sdk.hooks;
  const { fetchJSON } = sdk;
  const h = React.createElement;
  const API = "/api/plugins/marketing_factory";

  function cx(...parts) { return parts.filter(Boolean).join(" "); }

  // ---------------------------------------------------------------------------
  // FactoryFloor — only visible while a campaign is actively running.
  // ---------------------------------------------------------------------------
  const AGENT_CARDS = [
    { key: "brand_memory", label: "Brand Memory", emoji: "🧠" },
    { key: "research",     label: "Trend Spotter", emoji: "🔭" },
    { key: "strategy",     label: "Strategist",    emoji: "📐" },
    { key: "copy",         label: "Copywriter",    emoji: "✍️" },
    { key: "image_gen",    label: "Photographer",  emoji: "📸" },
    { key: "safety",       label: "Safety Officer", emoji: "🛡️" },
  ];

  function useFactoryProgress() {
    const [agentStates, setAgentStates] = useState({});
    const [campaignRunning, setCampaignRunning] = useState(null);
    useEffect(() => {
      const es = new EventSource(`${API}/progress/stream`);
      es.onmessage = (ev) => {
        if (!ev.data) return;
        let payload;
        try { payload = JSON.parse(ev.data); } catch (e) { return; }
        const type = payload.type;
        if (type === "campaign.start") {
          setCampaignRunning({ app_slug: payload.app_slug, days: payload.days, started_at: payload.timestamp });
        } else if (type === "campaign.end") {
          setCampaignRunning(null);
        } else if (type === "agent.start" || type === "agent.end") {
          setAgentStates((prev) => ({
            ...prev,
            [payload.agent]: {
              state: type === "agent.start" ? "working" : "idle",
              detail: payload.detail || "",
              at: payload.timestamp,
            },
          }));
        }
      };
      es.onerror = () => { /* browser auto-reconnects */ };
      return () => es.close();
    }, []);
    return { agentStates, campaignRunning };
  }

  function FactoryFloor({ campaignRunning, agentStates }) {
    if (!campaignRunning) return null;
    return h("div", { className: "rounded-2xl border border-cyan-300/30 bg-cyan-300/5 p-4" },
      h("div", { className: "flex items-center justify-between mb-3" },
        h("div", null,
          h("div", { className: "text-xs uppercase tracking-[0.22em] text-cyan-200/70" }, "Factory is making content"),
          h("div", { className: "text-base font-semibold" }, `${campaignRunning.app_slug} · ${campaignRunning.days} days`)
        ),
        h("div", { className: "h-3 w-3 rounded-full bg-cyan-300 animate-pulse" })
      ),
      h("div", { className: "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2" },
        AGENT_CARDS.map((agent) => {
          const state = agentStates[agent.key];
          const working = state && state.state === "working";
          return h("div", {
            key: agent.key,
            className: cx("rounded-xl border p-2.5 transition", working ? "border-cyan-300/50 bg-cyan-300/10" : "border-midground/15 bg-background/40"),
          },
            h("div", { className: "flex items-center gap-2" },
              h("span", { className: cx("text-base", working ? "" : "opacity-50") }, agent.emoji),
              h("span", { className: "text-xs font-medium" }, agent.label),
              h("span", { className: cx("ml-auto h-2 w-2 rounded-full", working ? "bg-cyan-300 animate-pulse" : "bg-midground/25") })
            ),
            h("div", { className: cx("mt-1.5 text-[10px] leading-snug min-h-[28px]", working ? "text-cyan-100" : "text-midground/60") },
              state && state.detail ? state.detail : (working ? "working…" : "ready")
            )
          );
        })
      )
    );
  }

  // ---------------------------------------------------------------------------
  // PlatformPreview — renders a draft as it would appear on its channel
  // ---------------------------------------------------------------------------
  function PlatformPreview({ draft, app }) {
    const channel = draft.channel;
    const brandName = (app && app.name) || draft.app_slug;
    const handle = `@${(draft.app_slug || "brand").toLowerCase().replace(/[^a-z0-9_]/g, "")}`;
    const body = draft.body || "";
    const firstImage = (draft.images || []).find((img) => img && img.url);

    if (channel === "x") {
      return h("div", { className: "rounded-2xl border border-midground/20 bg-background/80 p-4" },
        h("div", { className: "flex items-start gap-3" },
          h("div", { className: "h-10 w-10 rounded-full bg-cyan-300/40 shrink-0" }),
          h("div", { className: "flex-1 min-w-0" },
            h("div", { className: "flex items-baseline gap-1 text-sm" },
              h("span", { className: "font-semibold" }, brandName),
              h("span", { className: "text-midground/60" }, handle),
              h("span", { className: "text-midground/60" }, "· now")
            ),
            h("p", { className: "mt-1 whitespace-pre-wrap text-sm leading-6" }, body),
            firstImage ? h("img", { src: firstImage.url, alt: "post image", loading: "lazy", className: "mt-2 w-full rounded-2xl border border-midground/15" }) : null
          )
        )
      );
    }
    if (channel === "instagram") {
      return h("div", { className: "rounded-xl border border-midground/20 bg-background/80 p-3" },
        h("div", { className: "flex items-center gap-2 mb-2" },
          h("div", { className: "h-8 w-8 rounded-full bg-gradient-to-br from-pink-400 via-red-500 to-yellow-400" }),
          h("span", { className: "text-sm font-semibold" }, brandName)
        ),
        firstImage ? h("img", { src: firstImage.url, alt: "post image", loading: "lazy", className: "w-full aspect-square rounded-lg border border-midground/15 object-cover mb-2" }) : null,
        h("div", { className: "flex gap-3 text-lg mb-1" }, "♡  ◌  ⤴"),
        h("p", { className: "text-sm leading-6 whitespace-pre-wrap" },
          h("span", { className: "font-semibold mr-1" }, brandName),
          body
        )
      );
    }
    if (channel === "tiktok") {
      return h("div", { className: "rounded-xl border border-midground/20 bg-black text-white p-3 relative" },
        firstImage ? h("img", { src: firstImage.url, alt: "tiktok thumb", loading: "lazy", className: "w-full aspect-[9/16] rounded-lg object-cover mb-2 opacity-90" }) : h("div", { className: "w-full aspect-[9/16] rounded-lg bg-midground/40 mb-2" }),
        h("div", { className: "absolute bottom-3 left-3 right-3 text-sm" },
          h("div", { className: "font-semibold" }, brandName),
          h("p", { className: "mt-1 whitespace-pre-wrap leading-5" }, body)
        )
      );
    }
    if (channel === "linkedin") {
      return h("div", { className: "rounded-lg border border-midground/20 bg-background/80 p-4" },
        h("div", { className: "flex items-start gap-3 mb-3" },
          h("div", { className: "h-12 w-12 rounded-full bg-blue-400/40 shrink-0" }),
          h("div", null,
            h("div", { className: "font-semibold text-sm" }, brandName),
            h("div", { className: "text-xs text-midground/60" }, app && app.positioning ? (app.positioning.length > 60 ? app.positioning.slice(0, 60) + "…" : app.positioning) : "Brand · Now")
          )
        ),
        h("p", { className: "text-sm leading-6 whitespace-pre-wrap" }, body)
      );
    }
    if (channel === "blog") {
      return h("article", { className: "rounded-lg border border-midground/20 bg-background/80 p-4" },
        h("pre", { className: "whitespace-pre-wrap text-xs leading-6 font-mono text-midground/90" }, body)
      );
    }
    if (channel === "email") {
      const lines = body.split("\n");
      const subjectLine = lines[0] && lines[0].toLowerCase().startsWith("subject:") ? lines[0] : null;
      const rest = subjectLine ? lines.slice(1).join("\n") : body;
      return h("div", { className: "rounded-lg border border-midground/20 bg-background/80 p-4" },
        subjectLine ? h("div", { className: "border-b border-midground/15 pb-2 mb-3 text-sm font-semibold" }, subjectLine) : null,
        h("div", { className: "text-xs text-midground/60 mb-3" }, `From: ${brandName} <hello@${draft.app_slug}.com>`),
        h("p", { className: "text-sm leading-6 whitespace-pre-wrap" }, rest)
      );
    }
    if (channel === "app_store") {
      return h("div", { className: "rounded-2xl border border-midground/20 bg-gradient-to-br from-cyan-900/20 to-purple-900/20 p-4" },
        h("div", { className: "text-[10px] uppercase tracking-[0.18em] text-midground/70" }, "App Store promotional text"),
        h("p", { className: "mt-2 text-base font-medium leading-snug" }, body)
      );
    }
    return h("div", { className: "rounded-lg border border-midground/20 bg-background/80 p-3 text-sm whitespace-pre-wrap" }, body);
  }

  // ---------------------------------------------------------------------------
  // DraftCard — minimal: channel pill, preview, 3 actions.
  // ---------------------------------------------------------------------------
  function DraftCard({ draft, app, busy, run }) {
    return h("article", { className: "rounded-2xl border border-midground/15 bg-background/40 p-4" },
      h("div", { className: "flex items-center justify-between gap-2 mb-3" },
        h("span", { className: "text-xs uppercase tracking-[0.18em] text-midground/70" }, draft.channel),
        h("span", { className: "text-xs text-midground/50" }, draft.created_at ? new Date(draft.created_at).toLocaleString() : "")
      ),
      h(PlatformPreview, { draft, app }),
      h("div", { className: "mt-4 flex flex-wrap gap-2" },
        h("button", {
          type: "button",
          disabled: !!busy,
          onClick: () => run(`approve ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: "approved" }) })),
          className: cx(
            "flex-1 min-w-[120px] min-h-[48px] rounded-xl border px-4 py-2 text-sm font-medium transition",
            busy ? "cursor-not-allowed border-midground/10 text-midground/35" : "border-emerald-300/40 bg-emerald-300/10 text-emerald-100 hover:bg-emerald-300/20"
          ),
        }, "✓ Approve"),
        h("button", {
          type: "button",
          disabled: !!busy,
          onClick: () => {
            const reason = window.prompt("Why are you rejecting this? Be specific — your reason teaches the factory what to avoid.\n\nLeave blank to cancel.");
            if (!reason || !reason.trim()) return;
            return run(`reject ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/reject`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: "dashboard", reason: reason.trim() }) }));
          },
          className: cx(
            "flex-1 min-w-[120px] min-h-[48px] rounded-xl border px-4 py-2 text-sm font-medium transition",
            busy ? "cursor-not-allowed border-midground/10 text-midground/35" : "border-red-400/40 text-red-200 hover:bg-red-500/10"
          ),
        }, "✗ Reject"),
        h("button", {
          type: "button",
          disabled: !!busy,
          onClick: () => run(`regen ${draft.id}`, () => fetchJSON(`${API}/drafts/${encodeURIComponent(draft.id)}/regenerate`, { method: "POST" })),
          className: cx(
            "flex-1 min-w-[120px] min-h-[48px] rounded-xl border px-4 py-2 text-sm font-medium transition",
            busy ? "cursor-not-allowed border-midground/10 text-midground/35" : "border-midground/20 text-foreground hover:bg-midground/10"
          ),
        }, "↻ New version")
      )
    );
  }

  // ---------------------------------------------------------------------------
  // SettingsDrawer — slide-out panel for everything that's not the main flow.
  // ---------------------------------------------------------------------------
  function SettingsDrawer({ open, onClose, data, busy, run, advanced, setAdvanced }) {
    if (!open) return null;
    const summary = data?.summary || {};
    const advisor = data?.advisor;
    const budgets = summary.budgets || {};
    const poll = summary.poll || {};
    const lastPollAt = poll.last_poll_at ? new Date(poll.last_poll_at).toLocaleString() : "never";

    return h("div", { className: "fixed inset-0 z-50 flex" },
      h("div", { onClick: onClose, className: "absolute inset-0 bg-black/50" }),
      h("aside", { className: "relative ml-auto w-full max-w-md h-full overflow-y-auto bg-background border-l border-midground/20 p-6 flex flex-col gap-5" },
        h("div", { className: "flex items-center justify-between" },
          h("h2", { className: "text-lg font-semibold" }, "Settings"),
          h("button", { onClick: onClose, className: "text-midground/60 hover:text-foreground text-xl" }, "×")
        ),

        h("section", null,
          h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60 mb-2" }, "Numbers"),
          h("div", { className: "grid grid-cols-2 gap-2 text-sm" },
            h("div", { className: "rounded-lg border border-midground/15 bg-background/40 p-2" }, h("div", { className: "text-midground/60 text-xs" }, "Apps"), h("div", { className: "text-lg font-semibold" }, summary.apps ?? 0)),
            h("div", { className: "rounded-lg border border-midground/15 bg-background/40 p-2" }, h("div", { className: "text-midground/60 text-xs" }, "Campaigns"), h("div", { className: "text-lg font-semibold" }, summary.campaigns ?? 0)),
            h("div", { className: "rounded-lg border border-midground/15 bg-background/40 p-2" }, h("div", { className: "text-midground/60 text-xs" }, "Drafts"), h("div", { className: "text-lg font-semibold" }, summary.drafts ?? 0)),
            h("div", { className: "rounded-lg border border-midground/15 bg-background/40 p-2" }, h("div", { className: "text-midground/60 text-xs" }, "Pending"), h("div", { className: "text-lg font-semibold" }, summary.pending_approvals ?? 0))
          )
        ),

        advisor && !advisor.healthy ? h("section", null,
          h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60 mb-2" }, "Things to know"),
          h("div", { className: "flex flex-col gap-2 text-xs" },
            (advisor.items || []).map((item, idx) => h("div", {
              key: idx,
              className: cx("rounded-lg border p-2", item.severity === "warning" ? "border-amber-300/30 bg-amber-300/5" : "border-blue-300/30 bg-blue-300/5"),
            },
              h("div", { className: "font-medium" }, item.message),
              h("div", { className: "mt-1 text-midground/70" }, `→ ${item.action}`)
            ))
          )
        ) : null,

        h("section", null,
          h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60 mb-2" }, "Brands"),
          h("div", { className: "flex flex-col gap-2" },
            (data?.apps || []).map((app) => h("div", { key: app.slug, className: "rounded-lg border border-midground/15 bg-background/40 p-3 text-sm" },
              h("div", { className: "flex items-center justify-between gap-2" },
                h("div", { className: "font-semibold" }, app.name || app.slug),
                h("span", { className: cx("text-[10px] rounded-full border px-2 py-0.5", app.auto_generate ? "border-cyan-300/40 text-cyan-100" : "border-midground/20 text-midground/70") }, app.auto_generate ? `auto-on (≤${app.auto_generate_threshold || 3})` : "auto-off")
              ),
              h("div", { className: "mt-1 text-xs text-midground/60" }, app.positioning || "—"),
              h("div", { className: "mt-2 flex flex-wrap gap-2" },
                h("button", {
                  onClick: () => run(`auto ${app.slug}`, () => fetchJSON(`${API}/apps/${encodeURIComponent(app.slug)}/auto-generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: !app.auto_generate }) })),
                  disabled: !!busy,
                  className: "text-[10px] underline text-midground/70 hover:text-cyan-200",
                }, app.auto_generate ? "turn auto-gen off" : "turn auto-gen on"),
                h("button", {
                  onClick: async () => {
                    try {
                      const result = await fetchJSON(`${API}/apps/${encodeURIComponent(app.slug)}/digest?days=7`);
                      if (navigator.clipboard && navigator.clipboard.writeText) {
                        await navigator.clipboard.writeText(result.markdown);
                        alert(`7-day digest for ${app.slug} copied to clipboard.`);
                      }
                    } catch (e) { alert(`Digest failed: ${e.message || e}`); }
                  },
                  disabled: !!busy,
                  className: "text-[10px] underline text-midground/70 hover:text-cyan-200",
                }, "copy weekly digest")
              )
            ))
          ),
          h("button", {
            onClick: () => {
              const slug = window.prompt("New app slug (lowercase, alphanumeric, dashes/underscores ok):");
              if (!slug || !slug.trim()) return;
              const name = window.prompt(`Display name for ${slug}:`);
              if (!name) return;
              const positioning = window.prompt("Brand positioning (one sentence):") || "";
              const icp = window.prompt("Who buys/uses this?") || "";
              const tone = window.prompt("Brand tone (e.g. cute warm playful):") || "";
              const cta = window.prompt("Default call-to-action:") || "";
              const channelsStr = window.prompt("Channels (comma-separated: x,instagram,tiktok,linkedin,blog,email,app_store):") || "";
              const channels = channelsStr.split(",").map((s) => s.trim()).filter(Boolean);
              if (!channels.length) { alert("At least one channel required."); return; }
              const pillarsStr = window.prompt("Content pillars (recurring themes, comma-separated):") || "";
              const linkStr = window.prompt("Primary link (URL):") || "";
              return run("add-app", () => fetchJSON(`${API}/apps`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
                slug: slug.trim(), name: name.trim(), positioning, icp, tone, cta,
                channels,
                content_pillars: pillarsStr.split(",").map((s) => s.trim()).filter(Boolean),
                forbidden_claims: [], claims: [], competitors: [], assets: [],
                links: linkStr ? [linkStr] : [],
              }) }));
            },
            disabled: !!busy,
            className: "mt-3 w-full rounded-lg border border-cyan-300/40 bg-cyan-300/10 text-cyan-100 px-3 py-2 text-sm",
          }, "+ Add new brand")
        ),

        h("section", null,
          h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60 mb-2" }, "Cost"),
          h("div", { className: "text-sm text-midground/80" },
            `${(budgets.spent_tokens_today || 0).toLocaleString()} tokens used today (${budgets.daily_tokens ? Math.round(100 * (budgets.spent_tokens_today || 0) / budgets.daily_tokens) : 0}% of cap)`
          ),
          (() => {
            const cost = summary.cost_estimate;
            if (!cost) return null;
            const total = cost.total_usd || 0;
            return h("div", { className: "mt-2 rounded-lg border border-midground/15 bg-background/40 p-2" },
              h("div", { className: "flex items-baseline gap-2" },
                h("span", { className: "text-base font-semibold" }, `$${total.toFixed(2)}`),
                h("span", { className: "text-[10px] text-midground/60" }, "API-equivalent today")
              ),
              h("div", { className: "mt-1 text-[10px] text-midground/60" }, cost.note),
              h("div", { className: "mt-2 grid grid-cols-3 gap-2 text-[10px]" },
                ["cheap", "mid", "premium"].map((r) => h("div", { key: r, className: "rounded border border-midground/15 p-1.5" },
                  h("div", { className: "uppercase tracking-[0.14em] text-midground/60" }, r),
                  h("div", { className: "mt-0.5 font-medium" }, `$${(cost.by_route_usd?.[r] || 0).toFixed(4)}`),
                  h("div", { className: "text-midground/50" }, `$${(cost.rates_usd_per_m?.[r] || 0).toFixed(2)}/M`)
                ))
              )
            );
          })()
        ),

        h("section", null,
          h("div", { className: "text-xs uppercase tracking-[0.18em] text-midground/60 mb-2" }, "Auto-publishing loop"),
          h("div", { className: "text-xs text-midground/70" }, `Last tick: ${lastPollAt} · ${poll.total_polls || 0} ticks total`),
          h("button", {
            onClick: () => run("poll", () => fetchJSON(`${API}/poll`, { method: "POST" })),
            disabled: !!busy,
            className: "mt-2 text-sm underline text-cyan-200",
          }, "Run poll now"),
          h("div", { className: "mt-2 rounded-lg border border-midground/20 bg-background/40 p-2 font-mono text-[10px] text-midground/70 break-all" },
            `hermes marketing-factory enable-poller`
          )
        ),

        h("section", null,
          h("label", { className: "flex items-center gap-2 text-sm" },
            h("input", {
              type: "checkbox",
              checked: !!advanced,
              onChange: (e) => setAdvanced(e.target.checked),
            }),
            h("span", null, "Advanced mode (show pills, filters, bulk actions on the main page)")
          )
        )
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Main page
  // ---------------------------------------------------------------------------
  function MarketingFactoryPage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState(null);
    const [error, setError] = useState(null);
    const [appSlug, setAppSlug] = useState("pupular");
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [advanced, setAdvanced] = useState(false);
    const { agentStates, campaignRunning } = useFactoryProgress();

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

    // Auto-refresh after a campaign finishes
    useEffect(() => {
      if (campaignRunning === null) {
        refresh().catch(() => {});
      }
    }, [campaignRunning, refresh]);

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

    const currentApp = useMemo(() => (data?.apps || []).find((a) => a.slug === appSlug), [data, appSlug]);

    const reviewableDrafts = useMemo(() => {
      const drafts = data?.drafts || [];
      return drafts
        .filter((draft) => draft.app_slug === appSlug && draft.status === "needs_review")
        .slice()
        .reverse();
    }, [data, appSlug]);

    if (loading) return h("div", { className: "p-6 text-sm text-midground" }, "Loading…");

    return h("div", { className: "mx-auto flex w-full max-w-4xl flex-col gap-6 p-6 text-foreground" },

      // Header: just title + settings gear
      h("header", { className: "flex items-center justify-between" },
        h("div", null,
          h("h1", { className: "text-2xl font-semibold" }, "Marketing Factory"),
          h("p", { className: "text-sm text-midground" }, "Approve good posts. Reject bad ones with a reason. The factory learns.")
        ),
        h("button", {
          onClick: () => setSettingsOpen(true),
          className: "rounded-xl border border-midground/20 px-3 py-2 text-sm hover:bg-midground/10",
        }, "⚙ Settings")
      ),

      // Error banner if any
      error ? h("div", { className: "rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-100" }, error) : null,

      // Factory floor — ONLY visible while a campaign is running
      h(FactoryFloor, { campaignRunning, agentStates }),

      // App picker + the ONE big button
      h("section", { className: "rounded-2xl border border-cyan-300/30 bg-cyan-300/5 p-6 flex flex-col gap-4" },
        h("div", { className: "flex items-center justify-between gap-3" },
          h("div", { className: "flex items-center gap-2" },
            h("span", { className: "text-xs uppercase tracking-[0.18em] text-cyan-200/70" }, "Brand"),
            h("select", {
              value: appSlug,
              onChange: (e) => setAppSlug(e.target.value),
              className: "rounded-lg border border-midground/20 bg-background px-3 py-2 text-base font-semibold",
            },
              (data?.apps || []).map((app) => h("option", { key: app.slug, value: app.slug }, app.name || app.slug))
            )
          ),
          h("div", { className: "text-xs text-midground/70" }, currentApp ? `${(currentApp.channels || []).length} channels` : "")
        ),
        h("button", {
          onClick: () => run("generate", () => fetchJSON(`${API}/campaigns/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ app_slug: appSlug, days: 7 }) })),
          disabled: !!busy || !!campaignRunning || !appSlug,
          className: cx(
            "w-full min-h-[80px] rounded-2xl border-2 px-6 py-4 text-xl font-semibold transition",
            (!!busy || !!campaignRunning) ? "cursor-not-allowed border-midground/15 text-midground/40 bg-background/40" : "border-cyan-300/50 bg-cyan-300/15 text-cyan-50 hover:bg-cyan-300/25"
          ),
        }, campaignRunning ? "Factory is making content…" : busy === "generate" ? "Starting…" : `+ Make new content for ${currentApp?.name || appSlug}`),
        h("p", { className: "text-xs text-midground/60 text-center" }, "Generates 7 days of posts across all channels for this brand. ~60-90 seconds.")
      ),

      // The feed: drafts that need review
      h("section", null,
        h("div", { className: "flex items-baseline justify-between mb-4" },
          h("h2", { className: "text-lg font-semibold" }, reviewableDrafts.length === 0 ? "No new posts to review" : `${reviewableDrafts.length} post${reviewableDrafts.length === 1 ? "" : "s"} ready for you`),
          reviewableDrafts.length > 0 ? h("div", { className: "text-xs text-midground/60" }, "Approve / Reject / New version") : null
        ),
        reviewableDrafts.length === 0
          ? h("div", { className: "rounded-2xl border border-dashed border-midground/20 p-8 text-center text-sm text-midground/60" },
              campaignRunning ? "The factory is working. Posts will appear here in a moment." : "Click the big button above to make some.")
          : h("div", { className: "flex flex-col gap-4" },
              reviewableDrafts.map((draft) => h(DraftCard, { key: draft.id, draft, app: currentApp, busy, run }))
            )
      ),

      // Advanced section — only shows when toggled in settings
      advanced ? h("details", { className: "rounded-2xl border border-midground/15 bg-background/40 p-3 text-sm" },
        h("summary", { className: "cursor-pointer text-midground/80" }, "Advanced — other statuses & batch actions"),
        h("div", { className: "mt-3 text-xs text-midground/70" },
          `Drafts by status: ${JSON.stringify(data?.draft_status_counts || {})}`
        )
      ) : null,

      h(SettingsDrawer, { open: settingsOpen, onClose: () => setSettingsOpen(false), data, busy, run, advanced, setAdvanced })
    );
  }

  registry.register("marketing_factory", MarketingFactoryPage);
})();
