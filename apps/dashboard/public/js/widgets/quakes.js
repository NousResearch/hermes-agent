// Seismic monitor: recent M2.5+ earthquakes (USGS, no key). Fits the
// intelligence-console aesthetic; magnitude drives the color.

import { h, clear, timeAgo } from "../utils.js";
import { openViewer } from "../viewer.js";

const magClass = (m) => m >= 6 ? "q-major" : m >= 4.5 ? "q-strong" : "q-light";

export default {
  type: "quakes",
  title: "Seismic Monitor",
  icon: "🌐",
  defaultSize: "m",

  render(body, ctx) {
    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "SCANNING…"));
      let data;
      try {
        data = await ctx.api.quakes();
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Seismic feed unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      const list = h("div.quake-list");
      for (const q of data.quakes.slice(0, 12)) {
        list.append(h("button.quake-row", {
          type: "button", onclick: () => q.url && openViewer({ url: q.url, title: `M${q.mag} — ${q.place}`, mode: "embed" }),
        },
          h("span.quake-mag", { class: `quake-mag ${magClass(q.mag)}` }, q.mag.toFixed(1)),
          h("div.quake-main", {},
            h("div.quake-place", {}, q.place),
            h("div.muted.small", {}, timeAgo(new Date(q.time).toISOString()),
              q.tsunami ? h("span.quake-tsunami", {}, " · TSUNAMI") : null)),
        ));
      }
      if (!data.quakes.length) list.append(h("div.muted.small", {}, "No significant quakes in the last day."));
      clear(body).append(list, h("div.muted.small.quake-note", {}, "M2.5+ · past 24h · USGS"));
    };

    let last = null;
    ctx.onSummarize(() => last && ({
      kind: "recent earthquakes",
      title: "Seismic activity (past 24h)",
      content: last.quakes.slice(0, 12).map((q) => `M${q.mag} — ${q.place}`).join("\n"),
    }));
    ctx._track = (d) => { last = d; };

    ctx.onRefresh(draw);
    draw();
    ctx.every(10 * 60_000, draw);
  },
};
