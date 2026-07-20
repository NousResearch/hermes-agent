// AI Daily Brief — one interesting thing per category each day: a trending repo,
// a fresh paper, an AI headline, a learning pick and a prompt tip. Picks are
// deterministic-by-day over the live pools (with offline sample fallback), so
// the brief is fresh daily and stable within a day.

import { h, clear } from "../utils.js";
import { viewerLink } from "../viewer.js";

const LEARN = [
  ["Anthropic Academy (Skilljar)", "Free official Claude courses.", "https://anthropic.skilljar.com/"],
  ["Prompt engineering guide", "Official prompting tutorial.", "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview"],
  ["Building effective agents", "Anthropic's agent design patterns.", "https://www.anthropic.com/engineering/building-effective-agents"],
  ["Anthropic Cookbook", "Runnable notebooks & patterns.", "https://github.com/anthropics/anthropic-cookbook"],
  ["Claude Code docs", "Workflows, config, IDE integrations.", "https://docs.claude.com/en/docs/claude-code/overview"],
];
const PROMPTS = [
  "Ask Claude to plan before coding: files it'll touch, approach, trade-offs — then implement with tests.",
  "For gnarly bugs: have Claude ask clarifying questions one at a time before proposing a fix.",
  "Give Claude the acceptance criteria up front; ask it to self-check against them at the end.",
  "Use extended thinking for hard reasoning; keep the system prompt stable to benefit from prompt caching.",
  "Have Claude review its own diff: rank issues by severity with failure scenarios and file:line.",
];

const dayIndex = () => {
  const n = new Date();
  return Math.floor((n - new Date(n.getFullYear(), 0, 0)) / 86400000);
};
const pick = (arr) => (arr && arr.length ? arr[dayIndex() % arr.length] : null);

export default {
  type: "aidaily",
  title: "AI Daily Brief",
  icon: "☀️",
  defaultSize: "l",

  render(body, ctx) {
    const card = (label, title, sub, link) => {
      const inner = h("a.aid-card", { href: link?.url || "#", target: "_blank", rel: "noopener noreferrer" },
        h("span.aid-label", {}, label),
        h("div.aid-title", {}, title),
        sub ? h("div.muted.small.aid-sub", {}, sub) : null);
      return link ? viewerLink(inner, link) : inner;
    };

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "COMPILING TODAY'S BRIEF…"));
      const [repos, papers, news] = await Promise.allSettled([
        ctx.api.repos("week"), ctx.api.papers("cs.AI"), ctx.api.aiNews("claude"),
      ]);
      const grid = h("div.aid-grid");

      const repo = repos.status === "fulfilled" ? pick(repos.value.repos) : null;
      if (repo) grid.append(card("★ REPO OF THE DAY", repo.name,
        `${repo.desc || ""}${repo.language ? " · " + repo.language : ""}`,
        { url: repo.url, title: repo.name, source: "GitHub", mode: "embed" }));

      const paper = papers.status === "fulfilled" ? pick(papers.value.papers) : null;
      if (paper) grid.append(card("★ PAPER OF THE DAY", paper.title, paper.authors,
        { url: paper.url, title: paper.title, source: "arXiv", mode: "embed" }));

      const headline = news.status === "fulfilled" ? pick(news.value.items) : null;
      if (headline) grid.append(card("★ AI HEADLINE", headline.title, headline.source,
        { url: headline.url, title: headline.title, source: headline.source, mode: "reader" }));

      const learn = pick(LEARN);
      grid.append(card("★ LEARN TODAY", learn[0], learn[1],
        { url: learn[2], title: learn[0], source: "Learn", mode: "embed" }));

      grid.append(h("div.aid-card.aid-prompt", {},
        h("span.aid-label", {}, "★ PROMPT TIP"),
        h("div.aid-prompt-text", {}, pick(PROMPTS))));

      const anySample = [repos, papers, news].some((r) => r.status === "fulfilled" && r.value.source === "sample");
      ctx.setBadge(anySample ? "sample" : null);
      clear(body).append(grid,
        h("div.muted.small.aid-note", {}, "Fresh daily · picks rotate at midnight."));
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(60 * 60_000, draw);
  },
};
