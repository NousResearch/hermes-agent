import { h, clear, sparkline, fmtPrice, toast } from "../utils.js";

export default {
  type: "markets",
  title: "Markets",
  icon: "📈",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const watchlist = () => store.state.markets?.ids
      || ["bitcoin", "ethereum", "solana", "dogecoin"];

    const removeAsset = (asset) => {
      store.update((state) => {
        const targets = [asset.name.toLowerCase(), asset.symbol.toLowerCase()];
        state.markets.ids = watchlist().filter((id) => !targets.includes(id.toLowerCase()));
      }, "markets");
      draw();
    };

    const addAsset = () => {
      const id = prompt(
        "CoinGecko asset id to watch (e.g. bitcoin, cardano, chainlink):");
      const clean = (id || "").trim().toLowerCase().replace(/[^a-z0-9-]/g, "");
      if (!clean) return;
      if (watchlist().length >= 15) { toast("Watchlist is full (15)", "error"); return; }
      if (watchlist().includes(clean)) { toast("Already on the watchlist"); return; }
      store.update((state) => {
        if (!state.markets) state.markets = { ids: [] };
        state.markets.ids = [...watchlist(), clean];
      }, "markets");
      draw();
    };

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "Loading prices…"));
      let data;
      try {
        data = await ctx.api.markets(watchlist());
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Markets unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data.assets);
      const editing = store.state.editMode;

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
            editing ? h("button.icon-btn", {
              type: "button", title: "Remove from watchlist",
              "aria-label": `Remove ${asset.name} from watchlist`,
              onclick: () => removeAsset(asset),
            }, "✕") : null,
          ),
        );
      }
      clear(body).append(rows,
        h("div.market-note-row", {},
          h("span.muted.small", {}, "24h change · 7-day trend"),
          h("button.link-btn", { type: "button", onclick: addAsset }, "+ watch asset"),
        ));
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

    ctx.onStore((topic) => { if (topic === "editMode") draw(); });
    ctx.onRefresh(draw);
    draw();
    ctx.every(3 * 60_000, draw);
  },
};
