// Weather alerts (US National Weather Service, no key). Shares the weather
// widget's saved locations; lists active advisories/watches/warnings for the
// active location, most severe first, with a "clear" state when there are none.

import { h, clear, timeAgo } from "../utils.js";

const toneClass = { down: "wa-severe", warn: "wa-warn", neutral: "wa-minor" };

function fmtWindow(a) {
  const parts = [];
  if (a.expires) parts.push(`expires ${timeAgo(a.expires)}`);
  if (a.sender) parts.push(a.sender);
  return parts.join(" · ");
}

export default {
  type: "alerts",
  title: "Weather Alerts",
  icon: "⚠️",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const locations = () => store.state.weather?.locations || [];
    const activeLoc = () => locations()[store.state.weather?.active] || null;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "CHECKING ALERTS…"));
      const loc = activeLoc();
      let data;
      try {
        data = loc
          ? await ctx.api.alerts(loc.lat, loc.lon, loc.name)
          : await ctx.api.alerts(-29.8587, 31.0218, "Durban");
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Alerts unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const where = data.location?.name || "";
      if (!data.alerts.length) {
        clear(body).append(
          h("div.wa-clear", {},
            h("div.wa-clear-mark", { "aria-hidden": "true" }, "✓"),
            h("div", {}, "No active alerts"),
            h("div.muted.small", {}, where)));
        return;
      }

      const cards = data.alerts.map((a) => h("div.wa-card", { class: `wa-card ${toneClass[a.tone] || "wa-minor"}` },
        h("div.wa-card-head", {},
          h("span.wa-event", {}, a.event),
          h("span.wa-sev", {}, a.severity)),
        a.headline ? h("div.wa-headline", {}, a.headline) : null,
        a.area ? h("div.muted.small.wa-area", {}, a.area) : null,
        h("div.muted.small.wa-window", {}, fmtWindow(a))));

      clear(body).append(
        h("div.muted.small.wa-loc", {}, `${data.alerts.length} active · ${where}`),
        h("div.wa-list", {}, cards));
    };

    ctx.onRefresh(draw);
    ctx.onStore((topic) => { if (topic === "weather" || topic === "replace") draw(); });
    draw();
    ctx.every(10 * 60_000, draw);
  },
};
