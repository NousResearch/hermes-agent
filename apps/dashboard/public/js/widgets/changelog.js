// Claude Changelog — latest releases across the Claude Code, SDK and MCP repos
// (GitHub Releases, no key). Opens release notes in the in-app viewer.

import { h, clear, timeAgo } from "../utils.js";
import { viewerLink } from "../viewer.js";

export default {
  type: "changelog",
  title: "Claude Changelog",
  icon: "📦",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "CHECKING RELEASES…"));
      let data;
      try {
        data = await ctx.api.changelog();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Changelog unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      const list = h("div.chg-list");
      for (const r of data.releases) {
        list.append(viewerLink(
          h("a.chg-item", { href: r.url, target: "_blank", rel: "noopener noreferrer" },
            h("div.chg-top", {},
              h("span.chg-product", {}, r.product),
              h("span.chg-tag", {}, r.tag)),
            r.name && r.name !== `${r.product} ${r.tag}` ? h("div.chg-name", {}, r.name) : null,
            r.notes ? h("div.muted.small.chg-notes", {}, r.notes) : null,
            r.published ? h("div.muted.small.chg-when", {}, timeAgo(r.published)) : null),
          { url: r.url, title: `${r.product} ${r.tag}`, source: "GitHub", mode: "embed" }));
      }
      clear(body).append(list);
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(60 * 60_000, draw);
  },
};
