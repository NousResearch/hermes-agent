// Universal converter: coin ↔ fiat ↔ coin. Reuses cached crypto USD prices and
// USD-based fiat rates (no new upstream). Everything is normalised to USD then
// re-expressed in the target unit, so any pair converts in one hop.

import { h, clear } from "../utils.js";

export default {
  type: "convert",
  title: "Converter",
  icon: "🔄",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    let amount = store.state.convert?.amount ?? 1;
    let from = store.state.convert?.from || "BTC";
    let to = store.state.convert?.to || "USD";

    const persist = () => store.update((s) => { s.convert = { amount, from, to }; }, "convert");

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "LOADING RATES…"));
      let data;
      try {
        data = await ctx.api.convert();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Rates unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.asOf ? null : "sample");

      // Build the unit table: value of one unit in USD.
      const units = [];
      const usdOf = {};
      for (const [sym, info] of Object.entries(data.coins || {})) {
        units.push({ code: sym, label: `${sym} · ${info.name}`, kind: "coin" });
        usdOf[sym] = info.usd;
      }
      for (const [cur, rate] of Object.entries(data.fiat || {})) {
        units.push({ code: cur, label: cur, kind: "fiat" });
        usdOf[cur] = 1 / rate; // fiat rate = units per USD ⇒ 1 unit = 1/rate USD
      }
      if (!units.some((u) => u.code === from)) from = units[0]?.code || "USD";
      if (!units.some((u) => u.code === to)) to = "USD";

      const sel = (val, onchange) => h("select.select", { onchange },
        units.map((u) => h("option", { value: u.code, selected: u.code === val }, u.label)));

      const result = h("div.cv-result");
      const paint = () => {
        clear(result);
        const fu = usdOf[from];
        const tu = usdOf[to];
        if (!fu || !tu) { result.append(h("span.muted", {}, "—")); return; }
        const out = (amount * fu) / tu;
        const digits = out >= 1 ? (out >= 1000 ? 2 : 4) : 8;
        result.append(
          h("span.cv-out", {}, out.toLocaleString(undefined, { maximumFractionDigits: digits })),
          h("span.cv-unit", {}, to),
          h("div.muted.small.cv-rate", {}, `1 ${from} = ${((usdOf[from] / usdOf[to])).toLocaleString(undefined, { maximumFractionDigits: 6 })} ${to}`),
        );
      };

      const amountInput = h("input.input.cv-amount", {
        type: "number", step: "any", min: "0", value: String(amount), "aria-label": "Amount",
        oninput: (ev) => { amount = parseFloat(ev.target.value) || 0; persist(); paint(); },
      });
      const fromSel = sel(from, (ev) => { from = ev.target.value; persist(); paint(); });
      const toSel = sel(to, (ev) => { to = ev.target.value; persist(); paint(); });
      const swap = h("button.btn.cv-swap", {
        title: "Swap", "aria-label": "Swap units",
        onclick: () => { [from, to] = [to, from]; persist(); draw(); },
      }, "⇅");

      paint();
      clear(body).append(
        h("div.cv-row", {}, amountInput, fromSel, swap, toSel),
        result,
        h("div.muted.small.cv-note", {}, `${Object.keys(data.coins || {}).length} coins · live USD cross-rates`),
      );
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
