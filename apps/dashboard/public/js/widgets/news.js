import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";
import { summarizeButton } from "../summarize.js";

const LABELS = {
  top: "Top", world: "World", tech: "Tech", business: "Business",
  science: "Science", sports: "Sports", entertainment: "Culture",
};

let topicsCache = Object.keys(LABELS);

function labelFor(topic) {
  return LABELS[topic]
    || topic.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default {
  type: "news",
  title: "News",
  icon: "📰",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;

    const draw = async () => {
      const active = store.state.news.topic;

      const tabs = h("div.tabs", { role: "tablist", "aria-label": "News topics" },
        topicsCache.map((key) =>
          h("button.tab", {
            type: "button",
            role: "tab",
            "aria-selected": String(key === active),
            onclick: () => {
              store.update((state) => { state.news.topic = key; }, "news");
              draw();
            },
          }, labelFor(key)),
        ),
      );

      const list = h("div.news-list", {}, h("div.widget-loading", {}, "Fetching headlines…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.news(active, 24);
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `News unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._trackItems?.(data.items);

      clear(list);
      if (!data.items.length) {
        list.append(h("div.muted", {}, "No stories right now."));
        return;
      }
      for (const item of data.items) {
        const anchor = h("a.news-item", {
            href: item.url, target: "_blank", rel: "noopener noreferrer",
          },
            h("div.news-title", {}, item.title),
            item.summary ? h("div.news-summary", {}, item.summary) : null,
            h("div.news-meta", {},
              h("span.news-source", {}, item.source),
              item.published ? h("span", {}, " · ", timeAgo(item.published)) : null,
              data.source === "live" ? h("span.muted", {}, " · ", hostOf(item.url)) : null,
              summarizeButton(() => ({
                kind: "news story",
                title: item.title,
                content: `${item.title}\n${item.summary || ""}\nSource: ${item.source} — ${item.url}`,
              }), { cls: "icon-btn sum-btn sum-inline", tip: "Summarize this story" }),
            ),
          );
        list.append(viewerLink(anchor, {
          url: item.url,
          title: item.title,
          summary: item.summary,
          source: item.source,
          mode: "reader",
        }));
      }
    };

    let lastItems = [];
    ctx.onSummarize(() => ({
      kind: "set of news headlines",
      title: `${store.state.news.topic} news`,
      content: lastItems.map((i) => `${i.title} — ${i.summary || ""} (${i.source})`).join("\n"),
    }));
    const trackItems = (items) => { lastItems = items; };
    ctx._trackItems = trackItems;

    const loadTopics = async () => {
      try {
        const config = await ctx.api.feeds();
        topicsCache = config.topics;
        if (!topicsCache.includes(store.state.news.topic)) {
          store.update((state) => { state.news.topic = "top"; }, "news");
        }
        draw();
      } catch { /* keep last known tabs */ }
    };

    ctx.onStore((topic) => { if (topic === "news-external") draw(); });
    window.addEventListener("hub:feeds-changed", loadTopics);
    ctx.onRefresh(draw);
    loadTopics();
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
