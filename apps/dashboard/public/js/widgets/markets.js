import { h, clear, sparkline, fmtPrice, toast } from "../utils.js";
import { openDetail } from "../detail.js";
import { candleChart, donut } from "../chart.js";

// Portfolio math from holdings {coinId:{amount,cost?}} × a priceById map.
function portfolioRows(holdings, priceById) {
  const rows = [];
  for (const [id, hold] of Object.entries(holdings || {})) {
    const amount = Number(hold?.amount) || 0;
    if (amount <= 0) continue;
    const price = priceById[id];
    if (price == null) continue;
    const value = amount * price;
    const cost = hold.cost != null ? Number(hold.cost) : null;
    rows.push({ id, amount, price, value, cost, pl: cost != null ? value - cost : null });
  }
  rows.sort((a, b) => b.value - a.value);
  return rows;
}

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

// Fear & Greed zone → color class (0-24 extreme fear … 75-100 extreme greed).
const fgZone = (v) => v >= 75 ? "fg-egreed" : v >= 55 ? "fg-greed"
  : v >= 45 ? "fg-neutral" : v >= 25 ? "fg-fear" : "fg-efear";

function globalBar(g) {
  const stat = (label, value) => h("div.gm-stat", {},
    h("span.muted.small", {}, label), h("span.gm-v", {}, value));
  const fg = g.fearGreed;
  return h("div.global-bar", {},
    stat("MCAP", fmtBig(g.marketCap)),
    stat("BTC DOM", g.btcDominance != null ? `${g.btcDominance.toFixed(1)}%` : "—"),
    stat("VOL 24H", fmtBig(g.volume)),
    stat("24H", g.change24h != null
      ? h("span", { class: g.change24h >= 0 ? "delta-up" : "delta-down" },
          `${g.change24h >= 0 ? "+" : ""}${g.change24h.toFixed(2)}%`) : "—"),
    fg ? h("div.gm-fg", { class: `gm-fg ${fgZone(fg.value)}`, title: `Fear & Greed: ${fg.label}` },
      h("span.gm-fg-v", {}, String(fg.value)),
      h("span.small", {}, fg.label.toUpperCase())) : null,
  );
}

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

    // --- holdings for this coin + portfolio summary ---
    const holdings = ctx.store.state.markets?.holdings || {};
    const held = holdings[coinId] || {};
    const priceById = Object.fromEntries((ctx._assets || []).map((a) => [a.id, a.price]));
    priceById[coinId] = detail.price;  // ensure the open coin has a price
    const amountInput = h("input.input.hold-input", {
      type: "number", step: "any", min: "0", placeholder: "amount",
      value: held.amount != null ? String(held.amount) : "",
      "aria-label": "amount held",
    });
    const costInput = h("input.input.hold-input", {
      type: "number", step: "any", min: "0", placeholder: "cost basis $ (optional)",
      value: held.cost != null ? String(held.cost) : "",
      "aria-label": "cost basis",
    });
    const saveHolding = () => {
      const amount = parseFloat(amountInput.value);
      const cost = parseFloat(costInput.value);
      ctx.store.update((s) => {
        if (!s.markets) s.markets = { ids: [] };
        if (!s.markets.holdings) s.markets.holdings = {};
        if (!amount || amount <= 0) delete s.markets.holdings[coinId];
        else s.markets.holdings[coinId] = Number.isFinite(cost) && cost > 0
          ? { amount, cost } : { amount };
      }, "markets");
      draw();
    };
    const heldValue = (Number(held.amount) || 0) * (detail.price || 0);
    const holdingBox = h("div.hold-box", {},
      h("div.muted.small", {}, "YOUR POSITION"),
      h("div.hold-form", {}, amountInput, costInput,
        h("button.btn.btn-primary", { type: "button", onclick: saveHolding }, "Save")),
      held.amount ? h("div.hold-value", {},
        `${held.amount} ${detail.symbol} = $${fmtPrice(heldValue)}`,
        held.cost != null ? h("span.hold-pl", {},
          " · P/L ", pct(held.cost > 0 ? ((heldValue - held.cost) / held.cost) * 100 : 0)) : null,
      ) : null,
    );

    const pf = portfolioRows(holdings, priceById);
    const pfTotal = pf.reduce((a, r) => a + r.value, 0);
    const pfCost = pf.reduce((a, r) => a + (r.cost || 0), 0);
    const pfSection = pf.length ? h("div.pf-section", {},
      h("div.muted.small", {}, "PORTFOLIO"),
      h("div.pf-body", {},
        donut(pf.map((r) => ({ value: r.value, label: r.id })), { size: 108, thickness: 15 }),
        h("div.pf-legend", {},
          h("div.pf-total", {}, `$${fmtPrice(pfTotal)}`),
          pfCost > 0 ? h("div.small", {}, "P/L ", pct(((pfTotal - pfCost) / pfCost) * 100)) : null,
          ...pf.slice(0, 6).map((r) => h("div.pf-row.small", {},
            h("span", {}, r.id), h("span.muted", {}, `$${fmtPrice(r.value)}`))),
        ),
      ),
    ) : null;

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
      holdingBox,
      pfSection,
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
      let data; let glob = null; let trending = null;
      try {
        [data, glob, trending] = await Promise.all([
          ctx.api.markets(watchlist()),
          ctx.api.cryptoGlobal().catch(() => null),
          ctx.api.cryptoTrending().catch(() => null),
        ]);
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Markets unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data.assets);
      ctx._assets = data.assets;  // shared with the detail drawer for portfolio math
      const editing = store.state.editMode;

      const priceById = Object.fromEntries(data.assets.map((a) => [a.id, a.price]));
      const pf = portfolioRows(store.state.markets?.holdings, priceById);
      const pfTotal = pf.reduce((a, r) => a + r.value, 0);
      const pfCost = pf.reduce((a, r) => a + (r.cost || 0), 0);

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
      const trendStrip = trending?.coins?.length
        ? h("div.trend-strip", {},
            h("span.muted.small.trend-label", {}, "🔥 TRENDING"),
            trending.coins.slice(0, 6).map((c) => h("button.trend-chip", {
              type: "button", title: `${c.name} — add to watchlist`,
              onclick: () => {
                if (watchlist().includes(c.id)) { toast("Already watching " + c.symbol); return; }
                if (watchlist().length >= 15) { toast("Watchlist is full (15)", "error"); return; }
                store.update((s) => {
                  if (!s.markets) s.markets = { ids: [] };
                  s.markets.ids = [...watchlist(), c.id];
                }, "markets");
                draw();
              },
            }, c.symbol)))
        : null;

      const pfLine = pf.length ? h("div.pf-line", {},
        h("span.muted.small", {}, "PORTFOLIO"),
        h("span.pf-line-v", {}, `$${fmtPrice(pfTotal)}`),
        pfCost > 0 ? pct(((pfTotal - pfCost) / pfCost) * 100) : null,
      ) : null;

      clear(body).append(
        glob ? globalBar(glob) : null,
        rows,
        pfLine,
        trendStrip,
        h("div.market-note-row", {},
          h("span.muted.small", {}, "24h change · 7-day trend · tap a coin for detail"),
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
