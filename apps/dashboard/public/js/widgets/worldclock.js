// World clock: several time zones at a glance. Pure local (Intl) — no network.
// Zones are user-configurable and persist in synced state.

import { h, clear, toast } from "../utils.js";

const DEFAULTS = [
  { label: "London", tz: "Europe/London" },
  { label: "New York", tz: "America/New_York" },
  { label: "San Francisco", tz: "America/Los_Angeles" },
  { label: "Tokyo", tz: "Asia/Tokyo" },
];

const fmtTime = (tz) => new Intl.DateTimeFormat(undefined, {
  timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: false,
}).format(new Date());
const fmtDay = (tz) => new Intl.DateTimeFormat(undefined, {
  timeZone: tz, weekday: "short", month: "short", day: "numeric",
}).format(new Date());
// Offset from local for a quick "+3h" style hint.
function offsetHint(tz) {
  const now = new Date();
  const local = new Date(now.toLocaleString("en-US"));
  const there = new Date(now.toLocaleString("en-US", { timeZone: tz }));
  const diff = Math.round((there - local) / 3600000);
  return diff === 0 ? "same" : `${diff > 0 ? "+" : ""}${diff}h`;
}

export default {
  type: "worldclock",
  title: "World Clock",
  icon: "🕔",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const zones = () => store.state.worldclock?.zones || DEFAULTS;

    const draw = () => {
      const editing = store.state.editMode;
      const list = h("div.wc-list");
      for (const z of zones()) {
        list.append(h("div.wc-row", {},
          h("div.wc-place", {},
            h("div.wc-label", {}, z.label),
            h("div.muted.small", {}, `${fmtDay(z.tz)} · ${offsetHint(z.tz)}`)),
          h("div.wc-time", {}, fmtTime(z.tz)),
          editing ? h("button.icon-btn", {
            type: "button", title: "Remove", "aria-label": `Remove ${z.label}`,
            onclick: () => {
              store.update((s) => { if (!s.worldclock) s.worldclock = { zones: [] }; s.worldclock.zones = zones().filter((x) => x.tz !== z.tz); }, "worldclock");
              draw();
            },
          }, "✕") : null,
        ));
      }
      clear(body).append(list, h("div.market-note-row", {},
        h("span.muted.small", {}, "local offsets"),
        h("button.link-btn", {
          type: "button",
          onclick: () => {
            const tz = prompt("IANA time zone (e.g. Europe/Paris, Asia/Dubai):");
            if (!tz?.trim()) return;
            try { fmtTime(tz.trim()); } catch { toast("Unknown time zone", "error"); return; }
            const label = tz.trim().split("/").pop().replace(/_/g, " ");
            store.update((s) => { if (!s.worldclock) s.worldclock = { zones: DEFAULTS.slice() }; s.worldclock.zones = [...zones(), { label, tz: tz.trim() }]; }, "worldclock");
            draw();
          },
        }, "+ add zone")));
    };

    ctx.onStore((topic) => { if (topic === "editMode" || topic === "replace") draw(); });
    draw();
    ctx.every(30_000, draw); // minute precision is plenty
  },
};
