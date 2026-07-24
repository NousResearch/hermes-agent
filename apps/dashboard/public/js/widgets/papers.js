// arXiv Papers — latest AI research (cs.AI / cs.CL / cs.LG) via the free arXiv
// API. Expandable abstracts; "Summarize" hands the abstract to the agent.

import { h, clear } from "../utils.js";
import { viewerLink } from "../viewer.js";
import { summarizeButton } from "../summarize.js";

const CATS = [["cs.AI", "AI"], ["cs.CL", "NLP"], ["cs.LG", "ML"]];

export default {
  type: "papers",
  title: "arXiv Papers",
  icon: "📄",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const cat = () => store.state.papers?.cat || "cs.AI";

    const draw = async () => {
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "arXiv category" },
        CATS.map(([c, label]) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(c === cat()),
          onclick: () => { store.update((s) => { s.papers = { cat: c }; }, "papers"); draw(); },
        }, label)));
      const list = h("div.paper-list", {}, h("div.widget-loading", {}, "FETCHING PAPERS…"));
      clear(body).append(tabs, list);

      let data;
      try {
        data = await ctx.api.papers(cat());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Papers unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      clear(list);
      for (const p of data.papers) {
        const abstract = h("div.paper-abstract", { hidden: true }, p.summary);
        list.append(h("div.paper-item", {},
          h("button.paper-head", {
            type: "button", "aria-expanded": "false",
            onclick: (ev) => { const hidden = abstract.hidden; abstract.hidden = !hidden;
              ev.currentTarget.setAttribute("aria-expanded", String(hidden)); },
          }, h("span.paper-title", {}, p.title),
            h("span.paper-caret", { "aria-hidden": "true" }, "▾")),
          h("div.muted.small.paper-meta", {},
            [p.authors, p.published].filter(Boolean).join(" · ")),
          abstract,
          h("div.paper-actions", {},
            viewerLink(h("a.link-btn.small", { href: p.url, target: "_blank", rel: "noopener noreferrer" }, "open ↗"),
              { url: p.url, title: p.title, source: "arXiv", mode: "embed" }),
            summarizeButton(() => ({ kind: "research abstract", title: p.title,
              content: `${p.title}\n${p.summary}` }), { cls: "icon-btn sum-btn sum-inline", tip: "Summarize" }))));
      }
    };

    ctx.onSummarize(() => ({ kind: "arXiv listing", title: `${cat()} papers`, content: "" }));
    ctx.onRefresh(draw);
    draw();
    ctx.every(3 * 60 * 60_000, draw);
  },
};
