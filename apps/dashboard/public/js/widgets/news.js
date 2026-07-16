import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";
import { summarizeButton } from "../summarize.js";
import { saveForLater, markRead, isRead, isSaved } from "../reading.js";

const LABELS = {
  top: "Top", world: "World", tech: "Tech", business: "Business",
  science: "Science", sports: "Sports", entertainment: "Culture", gaming: "Gaming",
  medicine: "Medicine",
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
    let query = "";        // client-side search filter (never refetches)

    const matches = (item) => {
      if (!query) return true;
      const hay = `${item.title} ${item.summary || ""} ${item.source}`.toLowerCase();
      return query.split(/\s+/).every((term) => hay.includes(term));
    };

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

      const search = h("input.input.news-search", {
        type: "search",
        placeholder: "Filter headlines…",
        "aria-label": "Filter headlines",
        value: query,
        oninput: (ev) => {
          query = ev.target.value.trim().toLowerCase();
          renderItems();
        },
      });

      const list = h("div.news-list", {
        onscroll: () => { lastScroll = list.scrollTop; },
      }, h("div.widget-loading", {}, "Fetching headlines…"));
      clear(body).append(tabs, search, list);

      let data;
      try {
        data = await ctx.api.news(active, 24);
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `News unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._trackItems?.(data.items);
      lastItems = data.items;

      // renderItems paints lastItems through the current filter — the search
      // box calls it directly, so typing never hits the network.
      renderItems = () => {
      clear(list);
      const items = lastItems.filter(matches);
      if (!lastItems.length) {
        list.append(h("div.muted", {}, "No stories right now."));
        return;
      }
      if (!items.length) {
        list.append(h("div.muted", {}, `No headlines match “${query}”.`));
        return;
      }
      for (const item of items) {
        const anchor = h("a.news-item", {
            href: item.url, target: "_blank", rel: "noopener noreferrer",
            class: isRead(item.url) ? "news-item news-read" : "news-item",
          },
            h("div.news-title", {}, item.title),
            item.summary ? h("div.news-summary", {}, item.summary) : null,
            h("div.news-meta", {},
              h("span.news-source", {}, item.source),
              item.published ? h("span", {}, " · ", timeAgo(item.published)) : null,
              data.source === "live" ? h("span.muted", {}, " · ", hostOf(item.url)) : null,
              h("button.icon-btn.sum-inline.bookmark-btn", {
                type: "button",
                title: "Save to reading list",
                "aria-label": `Save “${item.title}” to reading list`,
                class: isSaved(item.url)
                  ? "icon-btn sum-inline bookmark-btn bookmark-saved"
                  : "icon-btn sum-inline bookmark-btn",
                onclick: (ev) => {
                  ev.preventDefault();
                  ev.stopPropagation();
                  saveForLater(item);
                  ev.currentTarget.classList.add("bookmark-saved");
                },
              }, "🔖"),
              summarizeButton(() => ({
                kind: "news story",
                title: item.title,
                content: `${item.title}\n${item.summary || ""}\nSource: ${item.source} — ${item.url}`,
              }), { cls: "icon-btn sum-btn sum-inline", tip: "Summarize this story" }),
            ),
          );
        anchor.addEventListener("click", () => {
          markRead(item.url);
          anchor.classList.add("news-read");
        });
        list.append(viewerLink(anchor, {
          url: item.url,
          title: item.title,
          summary: item.summary,
          source: item.source,
          mode: "reader",
        }));
      }
      list.scrollTop = lastScroll; // keep the reading position across redraws
      };
      renderItems();
    };

    let renderItems = () => {};  // reassigned by draw() once data is fetched
    let lastItems = [];
    let lastScroll = 0;
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

    ctx.onStore((topic) => {
      if (topic === "news-external") draw();
      // read/saved state changed elsewhere (reading widget, another device)
      if (topic === "reading" || topic === "replace") draw();
    });
    window.addEventListener("hub:feeds-changed", loadTopics);
    ctx.onRefresh(draw);
    loadTopics();
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
