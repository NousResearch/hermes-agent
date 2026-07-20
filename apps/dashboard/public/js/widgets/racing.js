// Motorsport — race calendar & results across F1, MotoGP, NASCAR and IndyCar
// (ESPN public scoreboard, no key). Each race shows status, circuit and — once
// finished — the podium. Sample fallback offline.

import { h, clear } from "../utils.js";

const SERIES = [
  ["f1", "F1"], ["motogp", "MotoGP"], ["nascar", "NASCAR"], ["indycar", "IndyCar"],
];

const stateChip = (state) => {
  if (state === "in") return h("span.score-chip.score-live", {}, "● LIVE");
  if (state === "post") return h("span.score-chip.score-final", {}, "RESULT");
  return h("span.score-chip.score-pre", {}, "UPCOMING");
};

export default {
  type: "racing",
  title: "Motorsport",
  icon: "🏁",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const series = () => store.state.racing?.series || "f1";

    const draw = async () => {
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "Series" },
        SERIES.map(([s, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(s === series()),
          onclick: () => { store.update((st) => { st.racing = { series: s }; }, "racing"); draw(); },
        }, label)));
      const list = h("div.race-list", {}, h("div.widget-loading", {}, "LOADING GRID…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.racing(series());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Motorsport unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      clear(list);
      if (!data.races.length) {
        list.append(h("div.muted.small", {}, "No races on the calendar right now."));
        return;
      }
      for (const r of data.races) {
        const podium = (r.top || []).length
          ? h("ol.race-podium", {}, r.top.map((name, i) =>
            h("li.race-pos", { class: `race-pos p${i + 1}` },
              h("span.race-medal", {}, ["🥇", "🥈", "🥉"][i] || `${i + 1}.`),
              h("span.race-driver", {}, name))))
          : null;
        list.append(h("div.race-card", { class: r.state === "in" ? "race-card race-card-live" : "race-card" },
          h("div.race-top", {},
            h("span.race-name", {}, r.name),
            stateChip(r.state)),
          r.circuit ? h("div.muted.small.race-circuit", {}, r.circuit) : null,
          h("div.muted.small.race-status", {}, r.status),
          podium));
      }
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
