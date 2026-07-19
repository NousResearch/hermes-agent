// Drug reference: official FDA label text for a medication (openFDA, no key).
// Reference only — not clinical advice. Sections (indications, dosage, warnings,
// contraindications, interactions, adverse reactions) are collapsible.

import { h, clear } from "../utils.js";

const toneFor = (label) => /warning|contraindication|boxed/i.test(label) ? "drug-warn"
  : /interaction|adverse/i.test(label) ? "drug-caution" : "drug-info";

export default {
  type: "drug",
  title: "Drug Reference",
  icon: "💊",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const query = () => store.state.drug?.query || "metformin";

    const draw = async () => {
      const input = h("input.input.drug-search", {
        type: "search", value: query(), placeholder: "Drug name (brand or generic)…",
        "aria-label": "Drug name",
        onkeydown: (ev) => { if (ev.key === "Enter") run(ev.target.value); },
      });
      const go = h("button.btn.drug-go", {
        type: "button", onclick: () => run(input.value),
      }, "Look up");
      const result = h("div.drug-result", {}, h("div.widget-loading", {}, "LOADING LABEL…"));
      clear(body).append(h("div.drug-head", {}, input, go), result);
      await load(result);
    };

    const run = (q) => {
      const term = (q || "").trim();
      if (!term) return;
      store.update((s) => { s.drug = { query: term }; }, "drug");
      draw();
    };

    const load = async (result) => {
      let data;
      try {
        data = await ctx.api.drug(query());
      } catch (err) {
        clear(result).append(h("div.widget-error", {}, `Lookup failed: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      const drug = data.drug;
      if (!drug) {
        clear(result).append(h("div.muted.small", {}, `No label found for “${data.query}”.`));
        return;
      }
      const title = drug.generic || drug.brand || data.query;
      const sub = [drug.brand && drug.brand !== title ? drug.brand : "", drug.route, drug.manufacturer]
        .filter(Boolean).join(" · ");
      const sections = drug.sections.map((s, i) => {
        const open = i === 0;
        const panel = h("div.drug-panel", { hidden: !open }, s.text);
        const btn = h("button.drug-section-btn", {
          type: "button", "aria-expanded": String(open),
          class: `drug-section-btn ${toneFor(s.label)}`,
          onclick: (ev) => {
            const hidden = panel.hidden; panel.hidden = !hidden;
            ev.currentTarget.setAttribute("aria-expanded", String(hidden));
          },
        }, h("span", {}, s.label), h("span.drug-caret", { "aria-hidden": "true" }, "▾"));
        return h("div.drug-section", {}, btn, panel);
      });
      clear(result).append(
        h("div.drug-name", {}, title),
        sub ? h("div.muted.small.drug-sub", {}, sub) : null,
        h("div.drug-sections", {}, sections),
        h("div.muted.small.drug-note", {}, "Source: FDA label (openFDA) · reference only, not clinical advice."),
      );
    };

    ctx.onRefresh(draw);
    draw();
  },
};
