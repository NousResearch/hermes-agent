// Commodities, metals & rates — grouped spot prices for precious/industrial
// metals, energy and US Treasury yields (Stooq, no key, sample fallback).

import { h, clear } from "../utils.js";

const GROUPS = ["Metals", "Energy", "Rates"];
const fmtPrice = (n) => (Math.abs(n) >= 100 ? n.toFixed(1) : n.toFixed(2));
const fmtPct = (p) => (p == null ? "" : `${p >= 0 ? "+" : ""}${p.toFixed(2)}%`);

export default {
  type: "commodities",
  title: "Commodities & Rates",
  icon: "🛢️",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "PRICING MARKETS…"));
      let data;
      try {
        data = await ctx.api.commodities();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Commodities unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);

      const wrap = h("div.commod-wrap");
      for (const group of GROUPS) {
        const rows = data.assets.filter((a) => a.group === group);
        if (!rows.length) continue;
        wrap.append(h("div.commod-group-label", {}, group));
        for (const a of rows) {
          const up = a.changePct != null && a.changePct >= 0;
          wrap.append(h("div.commod-row", {},
            h("div.commod-name", {},
              h("span.commod-sym", {}, a.symbol),
              h("span.muted.small.commod-unit", {}, a.unit)),
            h("div.commod-figs", {},
              h("span.commod-price", {}, fmtPrice(a.price)),
              h("span", { class: `commod-chg ${a.changePct == null ? "" : up ? "up" : "down"}` },
                fmtPct(a.changePct)))));
        }
      }
      clear(body).append(wrap);
    };

    ctx.onRefresh(draw);
    draw();
    ctx.every(5 * 60_000, draw);
  },
};
