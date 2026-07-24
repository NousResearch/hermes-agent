// Live sports scoreboard. League tabs over ESPN's public scoreboard (proxied,
// no key). Live games surface a status chip and refresh on a short interval.
// A ★ tab collects followed teams: their recent/next fixtures (with an
// add-to-calendar affordance) and a team-specific news feed.

import { h, clear, uid, timeAgo } from "../utils.js";
import { viewerLink } from "../viewer.js";

const LEAGUES = [
  ["nba", "NBA"], ["nfl", "NFL"], ["mlb", "MLB"], ["nhl", "NHL"],
  ["epl", "EPL"], ["laliga", "La Liga"], ["seriea", "Serie A"],
  ["bundesliga", "Bundesliga"], ["ligue1", "Ligue 1"], ["ucl", "UCL"],
  ["psl", "PSL"], ["mls", "MLS"],
  ["urc", "URC"], ["rugbyc", "Rugby C'ship"], ["cricket", "Cricket"],
  ["mma", "MMA"], ["atp", "Tennis ATP"], ["wta", "Tennis WTA"],
];

const followedTeams = (store) => store.state.sports?.teams || [];
const isFollowed = (store, league, abbr) =>
  followedTeams(store).some((t) => t.league === league && t.abbr === abbr);

function toggleFollow(store, team) {
  store.update((s) => {
    if (!s.sports) s.sports = { league: "nba", teams: [] };
    if (!s.sports.teams) s.sports.teams = [];
    const at = s.sports.teams.findIndex((t) => t.league === team.league && t.abbr === team.abbr);
    if (at >= 0) s.sports.teams.splice(at, 1);
    else s.sports.teams.push(team);
  }, "sports");
}

// Detail window: full standings table for the active league (via §0.2 ⤢).
async function renderStandings(body, ctx) {
  const league = ctx.store.state.sports?.league || "nba";
  clear(body).append(h("div.widget-loading", {}, "LOADING STANDINGS…"));
  let data;
  try {
    data = await ctx.api.standings(league);
  } catch (err) {
    clear(body).append(h("div.widget-error", {}, `Standings unavailable: ${err.message}`));
    return;
  }
  const sections = [h("div.muted.small.stand-league", {}, `${league.toUpperCase()} STANDINGS`)];
  for (const group of data.groups) {
    const head = h("div.stand-row.stand-head", {},
      h("span.stand-rank", {}, "#"), h("span.stand-team", {}, group.name || "Team"),
      ...data.columns.map((c) => h("span.stand-stat", {}, c.label)));
    const rows = group.teams.map((t, i) => h("div.stand-row", {},
      h("span.stand-rank", {}, String(i + 1)),
      h("span.stand-team", {}, h("b", {}, t.abbr), " ", h("span.muted", {}, t.name)),
      ...data.columns.map((c) => h("span.stand-stat", {}, t.stats[c.key] ?? "—"))));
    sections.push(h("div.stand-group", {}, head, ...rows));
  }
  clear(body).append(...sections);
}

const stateChip = (game) => {
  if (game.state === "in") return h("span.score-chip.score-live", {}, "● LIVE");
  if (game.state === "post") return h("span.score-chip.score-final", {}, "FINAL");
  return h("span.score-chip.score-pre", {}, "SCHED");
};

function teamRow(team, live, ctx, league) {
  const { store } = ctx;
  const followed = isFollowed(store, league, team.abbr);
  const star = h("button.icon-btn.score-follow", {
    type: "button",
    title: followed ? `Unfollow ${team.abbr}` : `Follow ${team.abbr}`,
    "aria-label": followed ? `Unfollow ${team.name}` : `Follow ${team.name}`,
    class: followed ? "icon-btn score-follow score-followed" : "icon-btn score-follow",
    onclick: (ev) => {
      ev.stopPropagation();
      toggleFollow(store, { league, abbr: team.abbr, name: team.name });
      ev.currentTarget.classList.toggle("score-followed");
    },
  }, followed ? "★" : "☆");
  return h("div.score-team", { class: team.winner ? "score-team score-win" : "score-team" },
    star,
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
  detail: renderStandings,

  render(body, ctx) {
    const { store } = ctx;
    const activeLeague = () => store.state.sports?.league || "nba";
    // "teams" is a pseudo-league selecting the My Teams view.
    let view = activeLeague();

    const buildTabs = () => {
      const league = activeLeague();
      const showTeams = view === "teams";
      const tabs = LEAGUES.map(([key, label]) => h("button.tab", {
        type: "button", role: "tab", "aria-selected": String(!showTeams && key === league),
        onclick: () => {
          view = key;
          store.update((s) => { if (!s.sports) s.sports = {}; s.sports.league = key; }, "sports");
          draw();
        },
      }, label));
      tabs.push(h("button.tab.score-teams-tab", {
        type: "button", role: "tab", "aria-selected": String(showTeams),
        onclick: () => { view = "teams"; draw(); },
      }, `★ Teams${followedTeams(store).length ? ` (${followedTeams(store).length})` : ""}`));
      return h("div.tabs", { role: "tablist", "aria-label": "Leagues" }, tabs);
    };

    const addToCalendar = (game, btn) => {
      const date = (game.start || "").slice(0, 10);
      if (!date) return;
      const title = `${game.away.abbr} @ ${game.home.abbr}`;
      store.update((s) => {
        s.calendar.events.push({ id: uid(), date, title: `🏆 ${title}` });
      }, "calendar");
      btn.textContent = "✓ Added";
      btn.disabled = true;
    };

    const drawMyTeams = async (list) => {
      const teams = followedTeams(store);
      clear(list);
      if (!teams.length) {
        list.append(h("div.muted.small.score-teams-empty", {},
          "No teams followed yet. Tap ☆ next to a team on any league board to follow it."));
        return;
      }
      for (const team of teams) {
        const section = h("div.myteam", {},
          h("div.myteam-head", {},
            h("b", {}, team.abbr), " ", h("span.muted", {}, team.name),
            h("span.muted.small.myteam-league", {}, team.league.toUpperCase()),
            h("button.icon-btn.score-follow.score-followed", {
              type: "button", title: `Unfollow ${team.abbr}`, "aria-label": `Unfollow ${team.name}`,
              onclick: () => { toggleFollow(store, team); drawMyTeams(list); },
            }, "★")));
        const games = h("div.myteam-games", {}, h("div.widget-loading", {}, "LOADING…"));
        section.append(games);
        list.append(section);

        ctx.api.teamSchedule(team.league, team.abbr).then((data) => {
          clear(games);
          const recent = data.games.filter((g) => g.state === "post").slice(-2);
          const upcoming = data.games.filter((g) => g.state !== "post").slice(0, 3);
          if (!recent.length && !upcoming.length) {
            games.append(h("div.muted.small", {}, "No fixtures found.")); return;
          }
          for (const g of [...recent, ...upcoming]) {
            const upcomingGame = g.state !== "post";
            games.append(h("div.myteam-game", { class: upcomingGame ? "myteam-game upcoming" : "myteam-game" },
              stateChip(g),
              h("span.myteam-match", {},
                `${g.away.abbr} ${g.away.score ?? ""} @ ${g.home.abbr} ${g.home.score ?? ""}`),
              h("span.muted.small.myteam-when", {}, g.status),
              upcomingGame ? h("button.btn.btn-tiny.myteam-cal", {
                type: "button", title: "Add to calendar",
                onclick: (ev) => addToCalendar(g, ev.currentTarget),
              }, "+ Cal") : null,
            ));
          }
        }).catch((err) => {
          clear(games).append(h("div.widget-error.small", {}, `Schedule unavailable: ${err.message}`));
        });

        const news = h("div.myteam-news");
        section.append(news);
        ctx.api.teamNews(team.name).then((data) => {
          for (const item of (data.items || []).slice(0, 3)) {
            news.append(viewerLink(
              h("a.myteam-news-item", { href: item.url, target: "_blank", rel: "noopener noreferrer" },
                h("span.myteam-news-title", {}, item.title),
                item.published ? h("span.muted.small", {}, " · ", timeAgo(item.published)) : null),
              { url: item.url, title: item.title, source: item.source, mode: "reader" }));
          }
        }).catch(() => { /* team news is best-effort */ });
      }
    };

    const draw = async () => {
      const list = h("div.score-list", {}, h("div.widget-loading", {}, "LOADING…"));
      clear(body).append(buildTabs(), list);

      if (view === "teams") {
        ctx.setBadge(null);
        await drawMyTeams(list);
        return;
      }

      const league = activeLeague();
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
          h("div.score-teams", {}, teamRow(game.away, live, ctx, league), teamRow(game.home, live, ctx, league)),
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
    ctx.every(60_000, () => { if (view !== "teams") draw(); });
  },
};
