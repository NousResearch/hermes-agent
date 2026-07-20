// Repo Radar — trending GitHub repositories (created in a recent window, ranked
// by stars). Free GitHub search API, proxied + cached; opens repos in-app.

import { h, clear } from "../utils.js";
import { viewerLink } from "../viewer.js";

const WINDOWS = [["day", "Today"], ["week", "Week"], ["month", "Month"]];
const fmtStars = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

export default {
  type: "repos",
  title: "Repo Radar",
  icon: "🛰️",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const window = () => store.state.repos?.window || "week";

    const draw = async () => {
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "Window" },
        WINDOWS.map(([w, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(w === window()),
          onclick: () => { store.update((s) => { s.repos = { window: w }; }, "repos"); draw(); },
        }, label)));
      const list = h("div.repo-list", {}, h("div.widget-loading", {}, "SCANNING GITHUB…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.repos(window());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Repos unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      clear(list);
      for (const r of data.repos) {
        list.append(viewerLink(
          h("a.repo-item", { href: r.url, target: "_blank", rel: "noopener noreferrer" },
            h("div.repo-top", {},
              h("span.repo-name", {}, r.name),
              h("span.repo-stars", {}, "★ ", fmtStars(r.stars))),
            r.desc ? h("div.muted.small.repo-desc", {}, r.desc) : null,
            h("div.repo-meta", {},
              r.language ? h("span.repo-lang", {}, r.language) : null,
              ...(r.topics || []).map((t) => h("span.repo-topic", {}, t)))),
          { url: r.url, title: r.name, source: "GitHub", mode: "embed" }));
      }
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(60 * 60_000, draw);
  },
};
