// Marine conditions (Open-Meteo Marine API, no key). Shares the weather
// widget's saved locations. Shows significant wave height + sea state, swell,
// wind-wave and sea-surface temperature. Defaults to Cape Town.

import { h, clear } from "../utils.js";

const fmt = (v, d = 1) => (v == null ? "—" : (Math.round(v * 10 ** d) / 10 ** d).toLocaleString());
const stateTone = (s) => ({
  Calm: "up", Smooth: "up", Moderate: "neutral", Rough: "warn",
  "Very rough": "down", High: "down",
}[s] || "neutral");

export default {
  type: "marine",
  title: "Marine & Surf",
  icon: "🌊",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const locations = () => store.state.weather?.locations || [];
    const activeLoc = () => locations()[store.state.weather?.active] || null;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "READING SEA STATE…"));
      const loc = activeLoc();
      let data;
      try {
        data = loc
          ? await ctx.api.marine(loc.lat, loc.lon, loc.name)
          : await ctx.api.marine(-33.92, 18.42, "Cape Town");
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Marine data unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const gauge = h("div.mar-gauge", { class: `mar-gauge tone-${stateTone(data.seaState)}` },
        h("div.mar-wave", {}, fmt(data.waveHeight), h("span.mar-unit", {}, " m")),
        h("div.mar-state", {}, data.seaState || "—"),
        h("div.muted.small.mar-loc", {}, data.location?.name || ""));

      const cell = (label, value, unit) => h("div.mar-cell", {},
        h("span.mar-cell-label", {}, label),
        h("span.mar-cell-val", {}, value),
        unit ? h("span.muted.mar-cell-unit", {}, unit) : null);

      const grid = h("div.mar-grid", {},
        cell("Swell", fmt(data.swellHeight), " m"),
        cell("Swell period", fmt(data.swellPeriod), " s"),
        cell("Wave period", fmt(data.wavePeriod), " s"),
        cell("Direction", data.waveDirText || "—",
          data.waveDir != null ? ` ${Math.round(data.waveDir)}°` : ""),
        cell("Wind wave", fmt(data.windWaveHeight), " m"),
        cell("Sea temp", fmt(data.seaTemp), " °C"),
        cell("24h max", fmt(data.waveMax), " m"));

      clear(body).append(gauge, h("div.mar-section-label", {}, "CONDITIONS"), grid);
    };

    ctx.onRefresh(draw);
    ctx.onStore((topic) => { if (topic === "weather" || topic === "replace") draw(); });
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
