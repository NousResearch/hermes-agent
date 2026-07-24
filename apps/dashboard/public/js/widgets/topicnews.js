// Topic-pinned headline widgets — one per tab, each locked to a news topic so
// every page carries relevant news without touching the global News widget's
// selected topic. Thin wrappers over api.news(topic) with the in-app reader.

import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";

function makeTopicNews(type, title, icon, topic) {
  return {
    type, title, icon, defaultSize: "m",
    render(body, ctx) {
      const draw = async () => {
        clear(body).append(h("div.widget-loading", {}, "LOADING HEADLINES…"));
        let data;
        try {
          data = await ctx.api.news(topic, 12);
        } catch (err) {
          clear(body).append(h("div.widget-error", {}, `News unavailable: ${err.message}`));
          return;
        }
        ctx.setBadge(data.source === "sample" ? "sample" : null);
        const list = h("div.news-list");
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
        clear(body).append(list);
      };
      ctx.onRefresh(draw);
      draw();
      ctx.every(20 * 60_000, draw);
    },
  };
}

export const marketsnews = makeTopicNews("marketsnews", "Markets News", "📈", "finance");
export const sportsnews = makeTopicNews("sportsnews", "Sports News", "🏅", "sports");
export const worldnews = makeTopicNews("worldnews", "World News", "🌍", "world");
export const healthnews = makeTopicNews("healthnews", "Health News", "⚕️", "medicine");
