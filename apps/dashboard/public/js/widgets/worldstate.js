// State-of-the-world situation board: per-domain stability index (0-100)
// with level, explanation and the headline signals behind the number.

import { h, clear, timeAgo } from "../utils.js";
import { openViewer } from "../viewer.js";

const LEVEL_LABEL = {
  stable: "STABLE",
  watch: "WATCH",
  elevated: "ELEVATED",
  critical: "CRITICAL",
};

function levelChip(level) {
  return h("span.level-chip", { class: `level-chip level-${level}` }, LEVEL_LABEL[level] || level);
}

function scoreBar(score, level) {
  return h("div.score-bar", {
    role: "img",
    "aria-label": `stability index ${score} of 100`,
  },
    h("div.score-bar-fill", { class: `score-bar-fill level-bg-${level}`, style: { width: `${score}%` } }),
  );
}

export default {
  type: "worldstate",
  title: "State of the World",
  icon: "🌐",
  defaultSize: "l",

  render(body, ctx) {
    const expanded = new Set(["geopolitics"]);
    let lastData = null;

    // Fetch fresh data from the API (shows the loading state). Called on first
    // load, manual refresh and the periodic timer — NOT on expand/collapse.
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "COMPILING SITUATION BOARD…"));
      let data;
      try {
        data = await ctx.api.worldstate();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Situation board unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      paint(data);
    };

    // Render already-fetched data. Expand/collapse calls this directly so the
    // widget never blanks to a loading state or hits the network on a click.
    const paint = (data) => {
      const header = h("div.ws-overall", {},
        h("div.ws-dial", {},
          h("div.ws-score", {}, String(data.overall.score)),
          h("div.muted.small", {}, "GLOBAL INDEX"),
        ),
        h("div.ws-overall-meta", {},
          levelChip(data.overall.level),
          h("div.muted.small", {}, `COMPILED ${timeAgo(data.generated).toUpperCase()}`),
        ),
      );

      const rows = h("div.ws-domains");
      for (const domain of data.domains) {
        const isOpen = expanded.has(domain.id);
        const toggle = h("button.ws-row", {
          type: "button",
          "aria-expanded": String(isOpen),
          onclick: () => {
            isOpen ? expanded.delete(domain.id) : expanded.add(domain.id);
            paint(data);  // local re-render only — no refetch, no flash
          },
        },
          h("span.ws-caret", { "aria-hidden": "true" }, isOpen ? "▾" : "▸"),
          h("span.ws-name", {}, domain.name),
          scoreBar(domain.score, domain.level),
          h("span.ws-value", {}, String(domain.score)),
          levelChip(domain.level),
        );
        rows.append(toggle);

        if (isOpen) {
          const detail = h("div.ws-detail", {},
            h("p.ws-explanation", {}, domain.explanation),
          );
          if (domain.signals.length) {
            detail.append(
              h("div.muted.small.ws-signals-label", {}, "SIGNALS"),
              ...domain.signals.map((signal) =>
                h("button.ws-signal", {
                  type: "button",
                  title: `keywords: ${signal.keywords.join(", ")}`,
                  onclick: () => openViewer({
                    url: signal.url,
                    title: signal.headline,
                    source: signal.source,
                    mode: "reader",
                  }),
                },
                  h("span.ws-signal-delta", {
                    class: `ws-signal-delta ${signal.delta < 0 ? "delta-down" : "delta-up"}`,
                  }, signal.delta < 0 ? "▼" : "▲"),
                  h("span.ws-signal-headline", {}, signal.headline),
                  h("span.muted.small", {}, signal.source),
                ),
              ),
            );
          }
          rows.append(detail);
        }
      }

      clear(body).append(header, rows,
        h("p.muted.small.ws-method", {}, data.method));
    };

    ctx.onSummarize(() => lastData && ({
      kind: "world situation board",
      title: `Global index ${lastData.overall.score} (${lastData.overall.level})`,
      content: lastData.domains
        .map((d) => `${d.name}: ${d.score} ${d.level}. ${d.explanation} ` +
          d.signals.map((s) => s.headline).join("; "))
        .join("\n"),
    }));
    ctx._track = (data) => { lastData = data; };  // keep latest for summarize/expand

    ctx.onRefresh(draw);
    draw();
    ctx.every(10 * 60_000, draw);
  },
};
