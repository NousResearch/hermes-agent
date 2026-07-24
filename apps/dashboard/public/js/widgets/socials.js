// Socials hub — read-only, no-account feeds across Hacker News, Lobsters and
// Reddit (all proxied, no keys). Items open in the in-app viewer. The Reddit
// subreddit is user-configurable and persists in synced state.

import { h, clear } from "../utils.js";
import { openViewer } from "../viewer.js";

const NETWORKS = [["hn", "HN"], ["lobsters", "Lobsters"], ["reddit", "Reddit"]];
const BADGE = { "Hacker News": "HN", Lobsters: "LO", Reddit: "RE" };

export default {
  type: "socials",
  title: "Socials",
  icon: "💬",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const active = () => store.state.socials?.network || "hn";
    const sub = () => store.state.socials?.sub || "popular";

    const draw = async () => {
      const network = active();
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "Networks" },
        NETWORKS.map(([key, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(key === network),
          onclick: () => {
            store.update((s) => { if (!s.socials) s.socials = {}; s.socials.network = key; }, "socials");
            draw();
          },
        }, label)));

      const controls = network === "reddit"
        ? h("button.link-btn.social-sub", {
            type: "button", title: "Change subreddit",
            onclick: () => {
              const next = prompt("Subreddit to follow (without r/):", sub());
              const clean = (next || "").trim().replace(/[^A-Za-z0-9_]/g, "");
              if (!clean) return;
              store.update((s) => { if (!s.socials) s.socials = {}; s.socials.sub = clean; }, "socials");
              draw();
            },
          }, `r/${sub()} ▾`)
        : null;

      const list = h("div.social-list", {}, h("div.widget-loading", {}, "LOADING FEED…"));
      clear(body).append(h("div.social-head", {}, tabs, controls), list);

      let data;
      try {
        data = await ctx.api.social(network, network === "reddit" ? sub() : undefined);
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Feed unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      clear(list);
      for (const item of data.items) {
        list.append(h("button.social-item", {
          type: "button",
          onclick: () => openViewer({ url: item.url, title: item.title, source: item.source, mode: "reader" }),
        },
          h("span.social-badge", { "aria-hidden": "true" }, BADGE[item.source] || "•"),
          h("div.social-main", {},
            h("div.social-title", {}, item.title),
            h("div.social-meta.muted.small", {},
              item.author ? `${item.author} · ` : "",
              item.meta ? `${item.meta} · ` : "",
              `▲ ${item.score ?? 0} · 💬 ${item.comments ?? 0}`),
          ),
        ));
      }
    };

    let lastData = null;
    ctx.onSummarize(() => lastData && ({
      kind: "social feed",
      title: `${active()} feed`,
      content: lastData.items.map((i) => `${i.title} (▲${i.score} 💬${i.comments}) — ${i.source}`).join("\n"),
    }));
    ctx._track = (data) => { lastData = data; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
