// AI Radar — AI / Claude news across a few curated queries (Google News RSS,
// no key). Opens stories in the in-app reader.

import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";

const TOPICS = [["claude", "Claude"], ["agents", "Agents"], ["llm", "LLMs"], ["oss", "Open source"]];

export default {
  type: "ainews",
  title: "AI Radar",
  icon: "📡",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const topic = () => store.state.ainews?.topic || "claude";

    const draw = async () => {
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "AI topic" },
        TOPICS.map(([t, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(t === topic()),
          onclick: () => { store.update((s) => { s.ainews = { topic: t }; }, "ainews"); draw(); },
        }, label)));
      const list = h("div.news-list", {}, h("div.widget-loading", {}, "TUNING AI RADAR…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.aiNews(topic());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `AI news unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      clear(list);
      for (const item of data.items) {
        list.append(viewerLink(
          h("a.news-item", { href: item.url, target: "_blank", rel: "noopener noreferrer" },
            h("div.news-title", {}, item.title),
            h("div.news-meta", {},
              h("span.news-source", {}, item.source),
              item.published ? h("span", {}, " · ", timeAgo(item.published)) : null,
              data.source === "live" ? h("span.muted", {}, " · ", hostOf(item.url)) : null)),
          { url: item.url, title: item.title, source: item.source, mode: "reader" }));
      }
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
