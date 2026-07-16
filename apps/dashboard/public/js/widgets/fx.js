// Currency converter + live rates (Frankfurter / ECB, no key). Enter an amount
// and base currency; see it converted across a set of quote currencies.

import { h, clear } from "../utils.js";

const CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR"];

export default {
  type: "fx",
  title: "Currency",
  icon: "💱",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    let amount = store.state.fx?.amount ?? 100;
    let base = store.state.fx?.base || "USD";

    const persist = () => store.update((s) => { s.fx = { amount, base }; }, "fx");

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "FETCHING RATES…"));
      let data;
      try {
        data = await ctx.api.fx(base, CURRENCIES.filter((c) => c !== base).join(","));
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Rates unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const amountInput = h("input.input.fx-amount", {
        type: "number", step: "any", min: "0", value: String(amount), "aria-label": "Amount",
        oninput: (ev) => { amount = parseFloat(ev.target.value) || 0; persist(); repaint(); },
      });
      const baseSel = h("select.select", {
        "aria-label": "Base currency",
        onchange: (ev) => { base = ev.target.value; persist(); draw(); },
      }, CURRENCIES.map((c) => h("option", { value: c, selected: c === base }, c)));

      const rows = h("div.fx-rows");
      const repaint = () => {
        clear(rows);
        for (const [cur, rate] of Object.entries(data.rates)) {
          rows.append(h("div.fx-row", {},
            h("span.fx-cur", {}, cur),
            h("span.fx-val", {}, (amount * rate).toLocaleString(undefined, { maximumFractionDigits: 2 })),
            h("span.muted.small.fx-rate", {}, `@ ${rate}`)));
        }
      };
      repaint();
      clear(body).append(
        h("div.fx-head", {}, amountInput, baseSel),
        rows,
        h("div.muted.small.fx-note", {}, `ECB rates · ${data.date || ""}`),
      );
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(60 * 60_000, draw);
  },
};
