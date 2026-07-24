// Stocks / indices / FX watchlist on Stooq (no key). Rows show price + session
// change; clicking a row opens a detail window with a history candle chart and
// the shared technical indicators. Reuses chart.js + detail.js + indicators.

import { h, clear, fmtPrice, toast } from "../utils.js";
import { openDetail } from "../detail.js";
import { candleChart } from "../chart.js";

const DEFAULTS = ["^spx", "^ndq", "^dji", "aapl.us", "msft.us", "eurusd"];

const pctSpan = (v) => {
  if (v == null) return h("span.muted", {}, "—");
  const up = v >= 0;
  return h("span.market-change", { class: `market-change ${up ? "delta-up" : "delta-down"}` },
    `${up ? "▲" : "▼"} ${up ? "+" : ""}${v.toFixed(2)}%`);
};

async function renderStockDetail(body, ctx) {
  const symbol = ctx.detailSymbol || (ctx.store.state.stocks?.symbols || DEFAULTS)[0];
  clear(body).append(h("div.widget-loading", {}, "COMPILING…"));
  let hist;
  try {
    hist = await ctx.api.stocksHistory(symbol);
  } catch (err) {
    clear(body).append(h("div.widget-error", {}, `History unavailable: ${err.message}`));
    return;
  }
  const ov = hist.overlays || {};
  const overlays = [
    { points: ov.sma20, color: "var(--accent)", dash: "3 3" },
    { points: ov.sma50, color: "#c98500", dash: "3 3" },
  ].filter((o) => (o.points || []).some((v) => v != null));
  clear(body).append(
    h("div.coin-head", {}, h("div.coin-price", {}, symbol.toUpperCase().replace(".US", ""))),
    h("div.coin-chart-wrap", {}, candleChart(hist.candles, { width: 560, height: 220, overlays })),
    h("div.coin-signals-label.muted.small", {}, "TECHNICALS · informational only"),
    h("div.coin-signals", {}, (hist.signals || []).map((s) =>
      h("div.coin-signal", { class: `coin-signal tone-${s.tone}` },
        h("span.muted.small", {}, s.label), h("span", {}, s.value)))),
  );
}

const exportRef = {
  type: "stocks",
  title: "Stocks & FX",
  icon: "📉",
  defaultSize: "m",
  detail: renderStockDetail,

  render(body, ctx) {
    const { store } = ctx;
    const list = () => store.state.stocks?.symbols || DEFAULTS;

    const addSymbol = () => {
      const s = prompt("Stooq symbol (e.g. aapl.us, ^ndq, eurusd, nvda.us):");
      const clean = (s || "").trim().toLowerCase().replace(/[^a-z0-9.^-]/g, "");
      if (!clean) return;
      if (list().length >= 15) { toast("Watchlist is full (15)", "error"); return; }
      if (list().includes(clean)) { toast("Already watching " + clean); return; }
      store.update((st) => { if (!st.stocks) st.stocks = { symbols: [] }; st.stocks.symbols = [...list(), clean]; }, "stocks");
      draw();
    };
    const remove = (id) => {
      store.update((st) => { st.stocks.symbols = list().filter((s) => s !== id); }, "stocks");
      draw();
    };

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "Loading quotes…"));
      let data;
      try {
        data = await ctx.api.stocks(list().join(","));
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Quotes unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data.assets);
      const editing = store.state.editMode;

      const rows = h("div.market-rows", { role: "table", "aria-label": "Stocks watchlist" });
      for (const a of data.assets) {
        rows.append(h("button.market-row", {
          role: "row", type: "button", title: `${a.name} — chart`,
          onclick: () => { if (editing) return; ctx.detailSymbol = a.id; openDetail(exportRef, ctx); },
        },
          h("div.market-id", {}, h("div.market-symbol", {}, a.symbol), h("div.muted.small", {}, a.name)),
          h("div.market-price", {},
            h("div.market-value", {}, a.price >= 10 ? `$${fmtPrice(a.price)}` : a.price.toFixed(4)),
            pctSpan(a.changePct)),
          editing ? h("span.icon-btn", { title: "Remove", "aria-label": `Remove ${a.name}`,
            onclick: (ev) => { ev.stopPropagation(); remove(a.id); } }, "✕") : null,
        ));
      }
      clear(body).append(rows, h("div.market-note-row", {},
        h("span.muted.small", {}, "session change · tap for chart"),
        h("button.link-btn", { type: "button", onclick: addSymbol }, "+ add symbol")));
    };

    let lastAssets = [];
    ctx.onSummarize(() => ({
      kind: "stock & FX watchlist", title: "Stocks & FX",
      content: lastAssets.map((a) => `${a.name} (${a.symbol}): ${a.price} · ${a.changePct != null ? a.changePct.toFixed(2) + "%" : "—"}`).join("\n"),
    }));
    ctx._track = (assets) => { lastAssets = assets; };

    ctx.onStore((topic) => { if (topic === "editMode") draw(); });
    ctx.onRefresh(draw);
    draw();
    ctx.every(2 * 60_000, draw);
  },
};

export default exportRef;
