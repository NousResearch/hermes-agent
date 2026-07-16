import { h, clear, sparkline, fmtPrice, toast } from "../utils.js";
import { openDetail } from "../detail.js";
import { candleChart } from "../chart.js";

const CHART_RANGES = [["1", "1D"], ["7", "7D"], ["30", "30D"], ["90", "90D"], ["365", "1Y"]];

const fmtBig = (n) => {
  if (n == null) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  return `$${fmtPrice(n)}`;
};

const pct = (v) => {
  if (v == null) return h("span.muted", {}, "—");
  const up = v >= 0;
  return h("span", { class: up ? "delta-up" : "delta-down" },
    `${up ? "+" : ""}${v.toFixed(2)}%`);
};

// Shared detail renderer: a coin drawer with stats, a range-selectable candle
// chart (with SMA/Bollinger overlays) and the technical read-outs. Opened by a
// market-row click (ctx.detailCoin set) or the widget's ⤢ button (first coin).
async function renderCoinDetail(body, ctx) {
  const watchlist = ctx.store.state.markets?.ids || ["bitcoin"];
  let coinId = ctx.detailCoin || watchlist[0];
  let days = "30";

  const draw = async () => {
    clear(body).append(h("div.widget-loading", {}, "COMPILING…"));
    let detail; let chart;
    try {
      [detail, chart] = await Promise.all([ctx.api.cryptoCoin(coinId), ctx.api.cryptoChart(coinId, days)]);
    } catch (err) {
      clear(body).append(h("div.widget-error", {}, `Coin data unavailable: ${err.message}`));
      return;
    }
    const picker = h("select.select", {
      "aria-label": "Coin",
      onchange: (ev) => { coinId = ev.target.value; draw(); },
    }, watchlist.map((id) => h("option", { value: id, selected: id === coinId }, id)));

    const stat = (label, value) => h("div.coin-stat", {},
      h("div.muted.small", {}, label), h("div.coin-stat-v", {}, value));

    const ov = chart.overlays || {};
    const overlays = [
      { points: ov.sma20, color: "var(--accent)", dash: "3 3" },
      { points: ov.sma50, color: "#c98500", dash: "3 3" },
      { points: ov.bollUpper, color: "var(--muted)", width: 0.7 },
      { points: ov.bollLower, color: "var(--muted)", width: 0.7 },
    ].filter((o) => (o.points || []).some((v) => v != null));

    const rangeTabs = h("div.tabs", { role: "tablist" },
      CHART_RANGES.map(([d, label]) => h("button.tab", {
        type: "button", role: "tab", "aria-selected": String(d === days),
        onclick: () => { days = d; draw(); },
      }, label)));

    const chartWrap = h("div.coin-chart-wrap", {},
      candleChart(chart.candles, { width: 560, height: 220, overlays }));

    const signals = h("div.coin-signals", {},
      (chart.signals || []).map((s) => h("div.coin-signal", { class: `coin-signal tone-${s.tone}` },
        h("span.muted.small", {}, s.label), h("span", {}, s.value))));

    clear(body).append(
      h("div.coin-head", {},
        picker,
        h("div.coin-price", {}, `$${fmtPrice(detail.price ?? 0)}`),
        pct(detail.changes?.["24h"]),
        detail.rank ? h("span.coin-rank", {}, `RANK #${detail.rank}`) : null,
      ),
      rangeTabs,
      chartWrap,
      h("div.coin-signals-label.muted.small", {}, "TECHNICALS · informational only"),
      signals,
      h("div.coin-stats", {},
        stat("Market cap", fmtBig(detail.marketCap)),
        stat("24h volume", fmtBig(detail.volume)),
        stat("Circulating", detail.supply ? Math.round(detail.supply).toLocaleString() : "—"),
        stat("Max supply", detail.maxSupply ? Math.round(detail.maxSupply).toLocaleString() : "—"),
        stat("ATH", `${fmtBig(detail.ath)} (${detail.athChange != null ? detail.athChange.toFixed(0) : "—"}%)`),
        stat("ATL", `${fmtBig(detail.atl)} (${detail.atlChange != null ? "+" + detail.atlChange.toFixed(0) : "—"}%)`),
        stat("1h", pct(detail.changes?.["1h"])),
        stat("7d", pct(detail.changes?.["7d"])),
        stat("30d", pct(detail.changes?.["30d"])),
        stat("1y", pct(detail.changes?.["1y"])),
      ),
    );
  };
  await draw();
}

const exportRef = {
  type: "markets",
  title: "Markets",
  icon: "📈",
  defaultSize: "m",
  detail: renderCoinDetail,

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
          h("button.market-row", {
            role: "row", type: "button",
            title: `${asset.name} — details`,
            onclick: () => {
              if (editing) return;
              ctx.detailCoin = asset.id || asset.name.toLowerCase();
              openDetail(exportRef, ctx);
            },
          },
            h("div.market-id", {},
              h("div.market-symbol", {}, asset.symbol),
              h("div.muted.small", {}, asset.name),
            ),
            h("div.market-spark", { "aria-hidden": "true" }, sparkline(asset.spark)),
            h("div.market-price", {},
              h("div.market-value", {}, `$${fmtPrice(asset.price)}`),
              change,
            ),
            editing ? h("span.icon-btn", {
              title: "Remove from watchlist",
              "aria-label": `Remove ${asset.name} from watchlist`,
              onclick: (ev) => { ev.stopPropagation(); removeAsset(asset); },
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

export default exportRef;
