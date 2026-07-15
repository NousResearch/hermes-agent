// Reading list: stories bookmarked from the News widget, with read tracking.

import { h, clear, timeAgo, hostOf } from "../utils.js";
import { viewerLink } from "../viewer.js";
import { markRead, removeSaved, isRead } from "../reading.js";

export default {
  type: "reading",
  title: "Reading list",
  icon: "🔖",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;

    const draw = () => {
      const { items } = store.state.reading;
      clear(body);
      if (!items.length) {
        body.append(h("div.muted.small.reading-empty", {},
          "Nothing saved. Tap the 🔖 on any story to queue it here."));
        return;
      }
      const list = h("div.reading-list");
      for (const item of items) {
        const anchor = h("a.news-item.reading-item", {
          href: item.url, target: "_blank", rel: "noopener noreferrer",
          class: isRead(item.url) ? "news-item reading-item news-read" : "news-item reading-item",
        },
          h("div.news-title", {}, item.title),
          h("div.news-meta", {},
            item.source ? h("span.news-source", {}, item.source) : null,
            h("span", {}, ` · saved ${timeAgo(item.saved)}`),
            h("span.muted", {}, ` · ${hostOf(item.url)}`),
            isRead(item.url) ? h("span.reading-read-chip", {}, " · READ") : null,
          ),
        );
        anchor.addEventListener("click", () => {
          markRead(item.url);
          draw();
        });
        viewerLink(anchor, {
          url: item.url, title: item.title, summary: item.summary,
          source: item.source, mode: "reader",
        });
        list.append(h("div.reading-row", {},
          anchor,
          h("button.icon-btn", {
            type: "button", title: "Remove from reading list",
            "aria-label": `Remove ${item.title}`,
            onclick: () => { removeSaved(item.url); draw(); },
          }, "✕"),
        ));
      }
      body.append(list);

      const readCount = items.filter((i) => isRead(i.url)).length;
      if (readCount) {
        body.append(h("div.task-footer", {},
          h("span.muted.small", {}, `${readCount}/${items.length} read`),
          h("button.link-btn", {
            type: "button",
            onclick: () => {
              store.update((state) => {
                state.reading.items = state.reading.items.filter((i) => !isRead(i.url));
              }, "reading");
              draw();
            },
          }, "Clear read"),
        ));
      }
    };

    ctx.onSummarize(() => ({
      kind: "reading list",
      title: "Saved stories",
      content: store.state.reading.items
        .map((i) => `${i.title} — ${i.summary || ""} (${i.source})`).join("\n")
        || "Reading list is empty.",
    }));
    ctx.onStore((topic) => { if (topic === "reading") draw(); });
    draw();
  },
};
