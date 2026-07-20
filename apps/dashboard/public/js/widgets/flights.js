// Flights overhead (OpenSky Network, no key). Lists airborne aircraft within
// ~100 km of the active weather location, lowest first, with altitude, ground
// speed, heading and climb/descent state. Refreshes on a short interval.

import { h, clear } from "../utils.js";

const m2ft = (m) => (m == null ? null : Math.round(m * 3.28084));
const ms2kt = (ms) => (ms == null ? null : Math.round(ms * 1.94384));

function climbGlyph(vr) {
  if (vr == null || Math.abs(vr) < 0.5) return { g: "→", cls: "fl-level", t: "level" };
  return vr > 0 ? { g: "↑", cls: "fl-climb", t: "climbing" } : { g: "↓", cls: "fl-descend", t: "descending" };
}

export default {
  type: "flights",
  title: "Flights Overhead",
  icon: "✈️",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const locations = () => store.state.weather?.locations || [];
    const activeLoc = () => locations()[store.state.weather?.active] || null;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "SCANNING AIRSPACE…"));
      const loc = activeLoc();
      let data;
      try {
        data = loc
          ? await ctx.api.flights(loc.lat, loc.lon, loc.name)
          : await ctx.api.flights(-29.8587, 31.0218, "Durban");
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Flight data unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const where = data.location?.name || "";
      if (!data.flights.length) {
        clear(body).append(h("div.muted.small.fl-empty", {},
          `No aircraft overhead right now near ${where}.`));
        return;
      }

      const rows = data.flights.map((f) => {
        const climb = climbGlyph(f.verticalRate);
        return h("div.fl-row", {},
          h("span.fl-call", {}, f.callsign),
          h("span.fl-country.muted.small", {}, f.country),
          h("span.fl-alt", {}, m2ft(f.altitude) != null ? `${m2ft(f.altitude).toLocaleString()} ft` : "—"),
          h("span.fl-spd.muted.small", {}, ms2kt(f.velocity) != null ? `${ms2kt(f.velocity)} kt` : ""),
          h("span.fl-dir", { title: `heading ${f.dir}` }, f.dir || ""),
          h("span.fl-climb-glyph", { class: `fl-climb-glyph ${climb.cls}`, title: climb.t }, climb.g));
      });

      clear(body).append(
        h("div.muted.small.fl-head", {}, `${data.count} overhead · ${where}`),
        h("div.fl-list", {}, rows));
    };

    ctx.onRefresh(draw);
    ctx.onStore((topic) => { if (topic === "weather" || topic === "replace") draw(); });
    draw();
    ctx.every(60_000, draw);
  },
};
