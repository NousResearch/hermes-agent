// Clinical trials: recently updated studies for a query (ClinicalTrials.gov
// API v2, no key). Track new procedures, interventions and research activity.

import { h, clear } from "../utils.js";
import { openViewer } from "../viewer.js";

const statusClass = (s) => /RECRUIT/i.test(s) ? "trial-recruiting"
  : /COMPLETED/i.test(s) ? "trial-done" : "trial-other";

export default {
  type: "trials",
  title: "Clinical Trials",
  icon: "🔬",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const query = () => store.state.trials?.query || "South Africa";

    const draw = async () => {
      const head = h("div.pubmed-head", {},
        h("span.pubmed-query.muted.small", {}, `“${query()}”`),
        h("button.link-btn", {
          type: "button",
          onclick: () => {
            const q = prompt("Trials search (condition, intervention, location):", query());
            if (!q?.trim()) return;
            store.update((s) => { s.trials = { query: q.trim() }; }, "trials");
            draw();
          },
        }, "edit search"));
      const list = h("div.trials-list", {}, h("div.widget-loading", {}, "SEARCHING TRIALS…"));
      clear(body).append(head, list);

      let data;
      try {
        data = await ctx.api.trials(query());
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Trials unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      clear(list);
      if (!data.trials.length) {
        list.append(h("div.muted.small", {}, "No trials found for that search."));
        return;
      }
      for (const t of data.trials) {
        list.append(h("button.trial-item", {
          type: "button",
          onclick: () => openViewer({ url: t.url, title: t.title, source: t.nct, mode: "embed" }),
        },
          h("div.trial-title", {}, t.title),
          h("div.trial-meta", {},
            h("span.trial-status", { class: `trial-status ${statusClass(t.status)}` }, (t.status || "").replace(/_/g, " ")),
            h("span.muted.small", {}, [t.conditions, t.updated].filter(Boolean).join(" · "))),
        ));
      }
    };

    let last = null;
    ctx.onSummarize(() => last && ({
      kind: "clinical trials",
      title: `Trials: ${query()}`,
      content: last.trials.map((t) => `${t.title} (${t.status}) — ${t.conditions}`).join("\n"),
    }));
    ctx._track = (d) => { last = d; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(30 * 60_000, draw);
  },
};
