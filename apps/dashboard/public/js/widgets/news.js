import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";
import { summarizeButton } from "../summarize.js";
import { saveForLater, markRead, isRead, isSaved } from "../reading.js";

const LABELS = {
  top: "Top", world: "World", tech: "Tech", business: "Business",
  science: "Science", sports: "Sports", entertainment: "Culture", gaming: "Gaming",
  medicine: "Medicine", southafrica: "South Africa", africa: "Africa",
  ai: "AI", finance: "Finance",
};

let topicsCache = Object.keys(LABELS);

function labelFor(topic) {
  return LABELS[topic]
    || topic.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Per-source mute/pin, stored in synced state and applied client-side.
const mutedSources = (store) => store.state.news?.muted || [];
const pinnedSources = (store) => store.state.news?.pinned || [];
const isMuted = (store, src) => mutedSources(store).includes(src);
const isPinned = (store, src) => pinnedSources(store).includes(src);

function toggleSource(store, key, src) {
  store.update((s) => {
    if (!s.news[key]) s.news[key] = [];
    const at = s.news[key].indexOf(src);
    if (at >= 0) s.news[key].splice(at, 1);
    else s.news[key].push(src);
  }, "news");
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
      if (isMuted(store, item.source)) return false;
      if (!query) return true;
      const hay = `${item.title} ${item.summary || ""} ${item.source}`.toLowerCase();
      return query.split(/\s+/).every((term) => hay.includes(term));
    };

    // Pinned sources float to the top; order within each band is preserved.
    const byPinned = (items) => {
      const pinned = items.filter((i) => isPinned(store, i.source));
      const rest = items.filter((i) => !isPinned(store, i.source));
      return [...pinned, ...rest];
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
      const muted = mutedSources(store);
      if (muted.length) {
        list.append(h("div.news-muted-bar", {},
          h("span.muted.small", {}, "Muted:"),
          ...muted.map((src) => h("button.news-muted-chip", {
            type: "button", title: `Unmute ${src}`, "aria-label": `Unmute ${src}`,
            onclick: () => { toggleSource(store, "muted", src); renderItems(); },
          }, src, " ✕"))));
      }
      const items = byPinned(lastItems.filter(matches));
      if (!lastItems.length) {
        list.append(h("div.muted", {}, "No stories right now."));
        return;
      }
      if (!items.length) {
        list.append(h("div.muted", {},
          query ? `No headlines match “${query}”.` : "Every source here is muted."));
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
              h("button.icon-btn.sum-inline.news-pin", {
                type: "button",
                title: isPinned(store, item.source) ? `Unpin ${item.source}` : `Pin ${item.source} to top`,
                "aria-label": `Pin source ${item.source}`,
                class: isPinned(store, item.source) ? "icon-btn sum-inline news-pin news-pin-on" : "icon-btn sum-inline news-pin",
                onclick: (ev) => {
                  ev.preventDefault(); ev.stopPropagation();
                  toggleSource(store, "pinned", item.source);
                  renderItems();
                },
              }, "📌"),
              h("button.icon-btn.sum-inline.news-mute", {
                type: "button",
                title: `Mute ${item.source}`,
                "aria-label": `Mute source ${item.source}`,
                onclick: (ev) => {
                  ev.preventDefault(); ev.stopPropagation();
                  toggleSource(store, "muted", item.source);
                  renderItems();
                },
              }, "🔇"),
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

  // ⤢ large topic view: more items grouped by source, plus a cross-topic
  // search toggle that queries every configured topic server-side.
  detail(body, ctx) {
    const { store } = ctx;
    let allTopics = false;
    let query = "";

    const groupBySource = (items) => {
      const groups = new Map();
      for (const item of items) {
        if (!groups.has(item.source)) groups.set(item.source, []);
        groups.get(item.source).push(item);
      }
      return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
    };

    const matches = (item) => {
      if (!query) return true;
      const hay = `${item.title} ${item.summary || ""} ${item.source} ${item.topic || ""}`.toLowerCase();
      return query.split(/\s+/).every((term) => hay.includes(term));
    };

    let items = [];
    const grid = h("div.news-detail-grid");

    const paint = () => {
      clear(grid);
      const filtered = items.filter(matches);
      if (!filtered.length) {
        grid.append(h("div.muted", {}, query ? `No headlines match “${query}”.` : "No stories right now."));
        return;
      }
      for (const [source, group] of groupBySource(filtered)) {
        const col = h("section.news-detail-col", {},
          h("h4.news-detail-src", {}, source, h("span.muted.small", {}, ` · ${group.length}`)));
        for (const item of group) {
          col.append(viewerLink(
            h("a.news-detail-item", { href: item.url, target: "_blank", rel: "noopener noreferrer" },
              h("div.news-title", {}, item.title),
              allTopics && item.topic ? h("span.news-topic-tag", {}, labelFor(item.topic)) : null,
              item.published ? h("div.news-meta", {}, timeAgo(item.published)) : null,
            ),
            { url: item.url, title: item.title, summary: item.summary, source: item.source, mode: "reader" },
          ));
        }
        grid.append(col);
      }
    };

    const scope = h("span.detail-scope", {}, labelFor(store.state.news.topic).toUpperCase());

    const load = async () => {
      scope.textContent = allTopics ? "ALL TOPICS" : labelFor(store.state.news.topic).toUpperCase();
      clear(grid).append(h("div.widget-loading", {}, "Fetching headlines…"));
      try {
        const data = allTopics
          ? await ctx.api.newsAll(60)
          : await ctx.api.news(store.state.news.topic, 40);
        items = data.items || [];
      } catch (err) {
        clear(grid).append(h("div.widget-error", {}, `News unavailable: ${err.message}`));
        return;
      }
      paint();
    };

    const search = h("input.input", {
      type: "search", placeholder: "Search headlines…", "aria-label": "Search headlines",
      oninput: (ev) => { query = ev.target.value.trim().toLowerCase(); paint(); },
    });
    const allToggle = h("label.news-all-toggle", {},
      h("input", {
        type: "checkbox",
        onchange: (ev) => { allTopics = ev.target.checked; load(); },
      }),
      " Search all topics");

    clear(body).append(
      h("div.news-detail-head", {}, scope, search, allToggle),
      grid);
    load();
  },
};
