// Air quality + pollen (Open-Meteo Air Quality API, no key). Shares the weather
// widget's saved locations. Shows the US AQI with a category band, the main
// pollutants, and — where the upstream provides it — a pollen breakdown.

import { h, clear } from "../utils.js";

const toneClass = { up: "aq-good", neutral: "aq-mod", warn: "aq-warn", down: "aq-bad" };
const fmt = (v) => (v == null ? "—" : (Math.round(v * 10) / 10).toLocaleString());

export default {
  type: "air",
  title: "Air Quality",
  icon: "🌬️",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const locations = () => store.state.weather?.locations || [];
    const activeLoc = () => locations()[store.state.weather?.active] || null;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "READING AIR QUALITY…"));
      const loc = activeLoc();
      let data;
      try {
        data = loc
          ? await ctx.api.air(loc.lat, loc.lon, loc.name)
          : await ctx.api.air(-29.8587, 31.0218, "Durban");
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Air data unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const tone = toneClass[data.band?.tone] || "aq-mod";
      const gauge = h("div.aq-gauge", { class: `aq-gauge ${tone}` },
        h("div.aq-aqi", {}, data.aqi == null ? "—" : String(data.aqi)),
        h("div.aq-band", {}, data.band?.label || ""),
        h("div.muted.small.aq-loc", {}, data.location?.name || ""));

      const pollutants = h("div.aq-pollutants", {},
        data.pollutants.map((p) => h("div.aq-cell", {},
          h("span.aq-cell-label", {}, p.label),
          h("span.aq-cell-val", {}, fmt(p.value)),
          h("span.muted.aq-cell-unit", {}, p.unit))));

      const children = [gauge, h("div.aq-section-label", {}, "POLLUTANTS"), pollutants];

      if (data.pollen?.length) {
        children.push(h("div.aq-section-label", {}, "POLLEN"));
        children.push(h("div.aq-pollen", {},
          data.pollen.map((p) => h("div.aq-pollen-row", {},
            h("span.aq-pollen-name", {}, p.label),
            h("span.aq-pollen-level", { class: `aq-pollen-level pl-${(p.level || "").split(" ")[0].toLowerCase()}` }, p.level),
            h("span.muted.small", {}, `${fmt(p.value)} gr/m³`)))));
      } else {
        children.push(h("div.muted.small.aq-nopollen", {}, "Pollen data not available for this location."));
      }

      clear(body).append(...children);
    };

    ctx.onRefresh(draw);
    // the weather widget owns the location list; re-draw when it changes
    ctx.onStore((topic) => { if (topic === "weather" || topic === "replace") draw(); });
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
