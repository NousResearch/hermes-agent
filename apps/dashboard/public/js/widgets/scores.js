// Live sports scoreboard. League tabs over ESPN's public scoreboard (proxied,
// no key). Live games surface a status chip and refresh on a short interval.

import { h, clear } from "../utils.js";

const LEAGUES = [
  ["nba", "NBA"], ["nfl", "NFL"], ["mlb", "MLB"], ["nhl", "NHL"],
  ["epl", "EPL"], ["mls", "MLS"],
];

const stateChip = (game) => {
  if (game.state === "in") return h("span.score-chip.score-live", {}, "● LIVE");
  if (game.state === "post") return h("span.score-chip.score-final", {}, "FINAL");
  return h("span.score-chip.score-pre", {}, "SCHED");
};

function teamRow(team, live) {
  return h("div.score-team", { class: team.winner ? "score-team score-win" : "score-team" },
    h("span.score-abbr", {}, team.abbr),
    h("span.score-name.muted.small", {}, team.name),
    h("span.score-num", { class: live ? "score-num score-num-live" : "score-num" },
      team.score != null ? String(team.score) : "—"),
  );
}

export default {
  type: "scores",
  title: "Scores",
  icon: "🏆",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const activeLeague = () => store.state.sports?.league || "nba";

    const draw = async () => {
      const league = activeLeague();
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "Leagues" },
        LEAGUES.map(([key, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(key === league),
          onclick: () => {
            store.update((s) => { if (!s.sports) s.sports = {}; s.sports.league = key; }, "sports");
            draw();
          },
        }, label)));

      const list = h("div.score-list", {}, h("div.widget-loading", {}, "LOADING SCORES…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.scores(league);
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Scores unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      clear(list);
      if (!data.games.length) {
        list.append(h("div.muted.small", {}, "No games on the board right now."));
        return;
      }
      for (const game of data.games) {
        const live = game.state === "in";
        list.append(h("div.score-game", { class: live ? "score-game score-game-live" : "score-game" },
          h("div.score-teams", {}, teamRow(game.away, live), teamRow(game.home, live)),
          h("div.score-meta", {}, stateChip(game),
            h("span.muted.small", {}, live ? (game.clock || game.status) : game.status)),
        ));
      }
    };

    let lastData = null;
    ctx.onSummarize(() => lastData && ({
      kind: "sports scoreboard",
      title: `${activeLeague().toUpperCase()} scores`,
      content: lastData.games.map((g) =>
        `${g.away.abbr} ${g.away.score ?? ""} @ ${g.home.abbr} ${g.home.score ?? ""} (${g.status})`).join("\n"),
    }));
    ctx._track = (data) => { lastData = data; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(60_000, draw);  // server caches 60s; live boards stay fresh enough
  },
};
