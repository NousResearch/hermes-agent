import { h, clear, sparkline, fmtPrice } from "../utils.js";

export default {
  type: "markets",
  title: "Markets",
  icon: "📈",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "Loading prices…"));
      let data;
      try {
        data = await ctx.api.markets();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Markets unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data.assets);

      const rows = h("div.market-rows", { role: "table", "aria-label": "Market watchlist" });
      for (const asset of data.assets) {
        const up = asset.change24h >= 0;
        // Direction is carried by sign + arrow, color is reinforcement only.
        const change = h("span.market-change", {
          class: `market-change ${up ? "delta-up" : "delta-down"}`,
        }, `${up ? "▲" : "▼"} ${up ? "+" : ""}${asset.change24h.toFixed(2)}%`);

        rows.append(
          h("div.market-row", { role: "row" },
            h("div.market-id", {},
              h("div.market-symbol", {}, asset.symbol),
              h("div.muted.small", {}, asset.name),
            ),
            h("div.market-spark", { "aria-hidden": "true" }, sparkline(asset.spark)),
            h("div.market-price", {},
              h("div.market-value", {}, `$${fmtPrice(asset.price)}`),
              change,
            ),
          ),
        );
      }
      clear(body).append(rows,
        h("div.muted.small.market-note", {}, "24h change · 7-day trend"));
    };

    let lastAssets = [];
    ctx.onSummarize(() => ({
      kind: "market watchlist",
      title: "Markets",
      content: lastAssets
        .map((a) => `${a.name} (${a.symbol}): $${a.price} · ${a.change24h.toFixed(2)}% over 24h`)
        .join("\n"),
    }));
    ctx._track = (assets) => { lastAssets = assets; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(3 * 60_000, draw);
  },
};
