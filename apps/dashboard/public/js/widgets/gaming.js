// Gaming data: Epic free games (current + upcoming) and Steam deals. Both from
// no-key store APIs, proxied. Opens store pages in the in-app viewer (embed).

import { h, clear, timeAgo } from "../utils.js";
import { openViewer } from "../viewer.js";

const untilLabel = (iso) => {
  if (!iso) return "";
  const days = Math.round((new Date(iso) - Date.now()) / 86400000);
  if (days <= 0) return "ends soon";
  return `${days}d left`;
};

export default {
  type: "gaming",
  title: "Gaming",
  icon: "🎮",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "LOADING GAMES…"));
      let free; let deals;
      try {
        [free, deals] = await Promise.all([
          ctx.api.gamingFree().catch(() => null),
          ctx.api.gamingDeals().catch(() => null),
        ]);
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Gaming data unavailable: ${err.message}`));
        return;
      }
      const sample = free?.source === "sample" || deals?.source === "sample";
      ctx.setBadge(sample ? "sample" : null);
      ctx._track?.({ free, deals });

      const open = (url, title) => openViewer({ url, title, mode: "embed" });

      const sections = [];
      const current = free?.current || [];
      const upcoming = free?.upcoming || [];
      if (current.length || upcoming.length) {
        const strip = h("div.game-free-list");
        for (const g of current) {
          strip.append(h("button.game-free", { type: "button", onclick: () => open(g.url, g.title) },
            h("span.game-free-tag.free-now", {}, "FREE NOW"),
            h("span.game-title", {}, g.title),
            h("span.muted.small", {}, untilLabel(g.end))));
        }
        for (const g of upcoming) {
          strip.append(h("button.game-free", { type: "button", onclick: () => open(g.url, g.title) },
            h("span.game-free-tag.free-soon", {}, "SOON"),
            h("span.game-title", {}, g.title),
            h("span.muted.small", {}, g.start ? timeAgo(g.start).replace(" ago", "") : "")));
        }
        sections.push(h("div.game-section", {},
          h("div.muted.small.game-label", {}, "🎁 FREE ON EPIC"), strip));
      }

      const dealItems = deals?.deals || [];
      if (dealItems.length) {
        const grid = h("div.game-deals");
        for (const d of dealItems.slice(0, 8)) {
          grid.append(h("button.game-deal", { type: "button", title: d.name, onclick: () => open(d.url, d.name) },
            h("span.game-disc", {}, `-${d.discount}%`),
            h("span.game-deal-name", {}, d.name),
            h("span.game-deal-price", {}, `$${d.price}`)));
        }
        sections.push(h("div.game-section", {},
          h("div.muted.small.game-label", {}, "🏷️ STEAM DEALS"), grid));
      }

      if (!sections.length) {
        clear(body).append(h("div.muted.small", {}, "No gaming data right now."));
        return;
      }
      clear(body).append(...sections);
    };

    let last = null;
    ctx.onSummarize(() => last && ({
      kind: "gaming deals and free games",
      title: "Gaming",
      content: [
        ...(last.free?.current || []).map((g) => `FREE now: ${g.title}`),
        ...(last.deals?.deals || []).slice(0, 8).map((d) => `${d.name} -${d.discount}% ($${d.price})`),
      ].join("\n"),
    }));
    ctx._track = (d) => { last = d; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
