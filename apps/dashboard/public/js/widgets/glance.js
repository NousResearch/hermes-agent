// At-a-glance "Command" hero: the morning brief in one widget. Composes data
// already on the board — next event, top task, weather, biggest market mover,
// top headline and the world index — with no new upstreams.

import { h, clear, weatherInfo } from "../utils.js";

const PRIORITY_RANK = { high: 0, normal: 1, low: 2 };

function ymd(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function topTask(state) {
  const open = (state.tasks?.lists || []).flatMap((l) => l.items).filter((i) => !i.done);
  if (!open.length) return null;
  open.sort((a, b) => {
    const pa = PRIORITY_RANK[a.priority] ?? 1; const pb = PRIORITY_RANK[b.priority] ?? 1;
    if (pa !== pb) return pa - pb;
    return (a.due || "9999") < (b.due || "9999") ? -1 : 1;
  });
  return open[0];
}

function nextUp(state) {
  const today = ymd(new Date());
  const events = (state.calendar?.events || []).filter((e) => (e.date || "") >= today)
    .map((e) => ({ date: e.date, label: e.title, kind: "event" }));
  const due = (state.tasks?.lists || []).flatMap((l) => l.items)
    .filter((i) => !i.done && i.due && i.due >= today)
    .map((i) => ({ date: i.due, label: i.text, kind: "due" }));
  const all = [...events, ...due].sort((a, b) => a.date < b.date ? -1 : 1);
  return all[0] || null;
}

export default {
  type: "glance",
  title: "At a Glance",
  icon: "🛰️",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "COMPILING…"));
      const loc = (store.state.weather?.locations || [])[store.state.weather?.active] || null;
      const [weather, markets, news, world] = await Promise.all([
        (loc ? ctx.api.weather(loc.lat, loc.lon, loc.name) : ctx.api.weather(-29.8587, 31.0218, "Durban")).catch(() => null),
        ctx.api.markets(store.state.markets?.ids).catch(() => null),
        ctx.api.news("top", 1).catch(() => null),
        ctx.api.worldstate().catch(() => null),
      ]);
      const sample = [weather, markets, news, world].some((d) => d && d.source === "sample");
      ctx.setBadge(sample ? "sample" : null);

      const cell = (label, value, sub) => h("div.glance-cell", {},
        h("div.glance-label.muted.small", {}, label),
        h("div.glance-value", {}, value),
        sub ? h("div.glance-sub.muted.small", {}, sub) : null);

      const task = topTask(store.state);
      const up = nextUp(store.state);
      const mover = markets?.assets?.length
        ? [...markets.assets].sort((a, b) => Math.abs(b.change24h) - Math.abs(a.change24h))[0]
        : null;
      const wi = weather?.current ? weatherInfo(weather.current.code) : null;
      const headline = news?.items?.[0];

      clear(body).append(
        h("div.glance-grid", {},
          cell("NEXT UP", up ? up.label : "Nothing scheduled",
            up ? `${up.kind === "due" ? "task due" : "event"} · ${up.date}` : "clear ahead"),
          cell("TOP TASK", task ? task.text : "All clear",
            task ? (task.priority ? `${task.priority} priority` : "open") : "inbox zero"),
          cell("WEATHER", weather?.current ? `${wi.icon} ${Math.round(weather.current.temp)}°` : "—",
            weather?.location?.name || ""),
          cell("BIGGEST MOVER", mover ? `${mover.symbol} ${mover.change24h >= 0 ? "+" : ""}${mover.change24h.toFixed(1)}%` : "—",
            mover ? mover.name : ""),
          cell("WORLD INDEX", world?.overall ? String(world.overall.score) : "—",
            world?.overall ? world.overall.level.toUpperCase() : ""),
          cell("HEADLINE", headline ? headline.title : "—", headline ? headline.source : ""),
        ),
      );
    };

    ctx.onRefresh(draw);
    ctx.onStore((topic) => { if (["tasks", "tasks-external", "calendar", "replace"].includes(topic)) draw(); });
    draw();
    ctx.every(10 * 60_000, draw);
  },
};
