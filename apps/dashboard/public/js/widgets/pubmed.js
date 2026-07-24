// PubMed feed: latest publications for a configurable query (NCBI E-utilities,
// no key). Keeps a clinician up to date with new literature in their field.

import { h, clear, toast } from "../utils.js";
import { openViewer } from "../viewer.js";

export default {
  type: "pubmed",
  title: "PubMed",
  icon: "🧬",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const query = () => store.state.pubmed?.query || "South Africa medicine";

    const draw = async () => {
      const head = h("div.pubmed-head", {},
        h("span.pubmed-query.muted.small", {}, `“${query()}”`),
        h("button.link-btn", {
          type: "button", title: "Change search",
          onclick: () => {
            const q = prompt("PubMed search (e.g. your specialty, a condition, an author):", query());
            if (!q?.trim()) return;
            store.update((s) => { s.pubmed = { query: q.trim() }; }, "pubmed");
            draw();
          },
        }, "edit search"));
      const list = h("div.pubmed-list", {}, h("div.widget-loading", {}, "SEARCHING PUBMED…"));
      clear(body).append(head, list);

      let data;
      try {
        data = await ctx.api.pubmed(query());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `PubMed unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      clear(list);
      if (!data.articles.length) {
        list.append(h("div.muted.small", {}, "No recent articles for that search."));
        return;
      }
      for (const a of data.articles) {
        list.append(h("button.pubmed-item", {
          type: "button",
          onclick: () => openViewer({ url: a.url, title: a.title, source: a.journal, mode: "embed" }),
        },
          h("div.pubmed-title", {}, a.title),
          h("div.pubmed-meta.muted.small", {},
            [a.authors, a.journal, a.date].filter(Boolean).join(" · ")),
        ));
      }
    };

    let last = null;
    ctx.onSummarize(() => last && ({
      kind: "recent medical literature",
      title: `PubMed: ${query()}`,
      content: last.articles.map((a) => `${a.title} — ${a.journal} (${a.date})`).join("\n"),
    }));
    ctx._track = (d) => { last = d; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
