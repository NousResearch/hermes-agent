// Space weather (NOAA SWPC, no key): the planetary K-index with its
// geomagnetic-storm band, the 24h peak, an aurora outlook, and a small bar
// history of recent Kp readings (0–9 scale).

import { h, clear } from "../utils.js";

const toneClass = { up: "sw-quiet", neutral: "sw-unsettled", warn: "sw-storm", down: "sw-severe" };
// Kp bar colour ramps green → amber → red as activity climbs.
const kpColor = (kp) => (kp >= 5 ? "var(--status-critical)"
  : kp >= 4 ? "var(--status-elevated, orange)" : "var(--status-stable)");
const hourOf = (t) => (t || "").slice(11, 16);

export default {
  type: "space",
  title: "Space Weather",
  icon: "🛰️",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "READING SOLAR ACTIVITY…"));
      let data;
      try {
        data = await ctx.api.spaceweather();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Space weather unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const tone = toneClass[data.band?.tone] || "sw-unsettled";
      const gauge = h("div.sw-gauge", { class: `sw-gauge ${tone}` },
        h("div.sw-kp", {}, `Kp ${data.kp?.toFixed(1) ?? "—"}`),
        h("div.sw-band", {}, data.band?.label || ""),
        h("div.muted.small.sw-aurora", {}, data.aurora || ""));

      const bars = h("div.sw-bars", {},
        (data.series || []).map((s) => h("div.sw-bar-wrap", { title: `${hourOf(s.t)} · Kp ${s.kp.toFixed(1)}` },
          h("div.sw-bar", { style: `height:${Math.max(4, (s.kp / 9) * 100)}%;background:${kpColor(s.kp)}` }),
          h("span.muted.sw-bar-label", {}, hourOf(s.t)))));

      clear(body).append(
        gauge,
        h("div.sw-meta", {},
          h("span.muted.small", {}, `24h peak: `),
          h("span.sw-peak", {}, `Kp ${data.peak24h?.toFixed(1) ?? "—"}`)),
        h("div.sw-section-label", {}, "PLANETARY K-INDEX"),
        bars);
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
