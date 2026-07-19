// System — a live status panel for the agent (Jarvis Phase 3, Layer H).
// Surfaces what the router and permission gate are doing: active engine/tier,
// the tier line-up, deep-tier budget, permission split, and a feed of recent
// tool calls (name + tier + outcome), all from /api/assistant/status and
// /api/assistant/telemetry. Polls every few seconds; ↻ refreshes on demand.

import { h, clear } from "../utils.js";
import { openEvolve } from "../evolve.js";

const TIER_LABEL = { fast: "FAST", core: "CORE", deep: "DEEP" };

function ago(sec) {
  const d = Date.now() / 1000 - sec;
  if (d < 60) return `${Math.max(0, Math.round(d))}s`;
  if (d < 3600) return `${Math.round(d / 60)}m`;
  return `${Math.round(d / 3600)}h`;
}

export default {
  type: "system",
  title: "System",
  icon: "⌘",
  defaultSize: "m",

  render(body, ctx) {
    const { api } = ctx;
    let last = null;

    const toggleFreeze = async (frozen) => {
      try {
        await api.setKillswitch(frozen);
        load();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Could not change autonomy: ${err.message}`));
      }
    };

    const draw = (status, tel, ks, ev) => {
      const frozen = !!(ks && ks.frozen);
      const pending = ev ? ev.pending : 0;
      const rows = [];
      const engine = status.mode === "claude"
        ? (status.routing && !status.routing.pinned ? "CLAUDE · ROUTED" : `CLAUDE · ${status.model}`)
        : "LOCAL ENGINE";
      rows.push(h("div.sys-row", {}, h("span.sys-key", {}, "ENGINE"), h("span.sys-val", {}, engine)));

      if (status.routing) {
        const r = status.routing;
        for (const t of ["fast", "core", "deep"]) {
          rows.push(h("div.sys-row", {},
            h("span.sys-key", {}, TIER_LABEL[t]),
            h("span.sys-val.muted", {}, r.tiers[t])));
        }
        rows.push(h("div.sys-row", {},
          h("span.sys-key", {}, "DEEP BUDGET"),
          h("span.sys-val", {}, `${r.deep_calls_last_hour}/${r.deep_cap_per_hour} this hr`)));
      }

      const perms = status.permissions || {};
      const counts = Object.values(perms).reduce((a, t) => { a[t] = (a[t] || 0) + 1; return a; }, {});
      rows.push(h("div.sys-row", {},
        h("span.sys-key", {}, "TOOLS"),
        h("span.sys-val", {}, `${counts.auto || 0} auto · ${counts.confirm || 0} confirm`)));

      const s = tel.summary;
      rows.push(h("div.sys-row", {},
        h("span.sys-key", {}, "ACTIVITY"),
        h("span.sys-val", {},
          `${s.tool_calls} tool call${s.tool_calls === 1 ? "" : "s"} · ${s.denied} denied`
          + (s.escalations ? ` · ${s.escalations} escalated` : ""))));

      // self-evolution proposals awaiting review
      rows.push(h("div.sys-row", {},
        h("span.sys-key", {}, "PROPOSALS"),
        h("span.sys-val", {},
          h("span", { class: pending ? "sys-proposals sys-proposals-open" : "sys-proposals" },
            pending ? `${pending} pending` : "none"),
          h("button.link-btn.sys-review-btn", {
            type: "button", title: "Review agent proposals",
            onclick: () => openEvolve(),
          }, "review"))));

      // kill switch — one toggle freezes all autonomous behaviour
      rows.push(h("div.sys-row", {},
        h("span.sys-key", {}, "AUTONOMY"),
        h("span.sys-val", {},
          h("span", { class: frozen ? "sys-autonomy sys-autonomy-frozen" : "sys-autonomy" },
            frozen ? "FROZEN" : "ACTIVE"),
          h("button.link-btn.sys-freeze-btn", {
            type: "button",
            title: frozen ? "Resume automations" : "Freeze all autonomous behaviour",
            onclick: () => toggleFreeze(!frozen),
          }, frozen ? "resume" : "freeze"))));

      const toolEvents = tel.events.filter((e) => e.kind === "tool").slice(-6).reverse();
      const feed = h("div.sys-feed", {},
        toolEvents.length
          ? toolEvents.map((e) => h("div.sys-event", { class: e.ok ? "sys-event" : "sys-event sys-event-bad" },
            h("span.sys-event-mark", {}, e.ok ? "✓" : (e.approved === false ? "⊘" : "✗")),
            h("span.sys-event-name", {}, e.name),
            h("span.muted.small", {}, `${e.tier}${e.approved === false ? " · denied" : ""} · ${ago(e.at)}`)))
          : h("div.muted.small", {}, "No agent activity yet."));

      clear(body);
      if (frozen) {
        body.append(h("div.sys-frozen-banner", {}, "⛔ AUTONOMY FROZEN — automations paused"));
      }
      body.append(
        h("div.sys-grid", {}, ...rows),
        h("div.muted.small.sys-feed-head", {}, "RECENT TOOL CALLS"),
        feed);
    };

    const load = async () => {
      try {
        const [status, tel, ks, ev] = await Promise.all([
          api.assistantStatus(), api.telemetry(), api.killswitch(), api.evolve()]);
        last = { status, tel, ks, ev };
        draw(status, tel, ks, ev);
      } catch (err) {
        if (!last) clear(body).append(h("div.widget-error", {}, `System status unavailable: ${err.message}`));
      }
    };

    ctx.onRefresh(load);
    ctx.onSummarize(() => last && ({
      kind: "agent system status",
      title: "System",
      content: JSON.stringify({ mode: last.status.mode, routing: last.status.routing, activity: last.tel.summary }, null, 2),
    }));
    clear(body).append(h("div.widget-loading", {}, "READING SYSTEM STATUS…"));
    load();
    ctx.every(6000, load);
  },
};
