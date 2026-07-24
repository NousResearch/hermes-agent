// Podcasts: subscribe to show RSS feeds and play episodes inline. Feeds are
// parsed server-side into audio enclosures; playback uses a native <audio>
// element (the browser streams the audio URL directly).

import { h, clear, timeAgo, toast } from "../utils.js";

const DEFAULTS = [];

export default {
  type: "podcasts",
  title: "Podcasts",
  icon: "🎧",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const feeds = () => store.state.podcasts?.feeds || DEFAULTS;
    let active = store.state.podcasts?.active || (feeds()[0]?.url);

    const player = h("audio.pod-player", { controls: true, preload: "none" });

    const draw = async () => {
      const tabs = h("div.tabs", { role: "tablist", "aria-label": "Shows" },
        feeds().map((f) => h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(f.url === active),
          onclick: () => { active = f.url; store.update((s) => { if (!s.podcasts) s.podcasts = { feeds: [] }; s.podcasts.active = f.url; }, "podcasts"); draw(); },
        }, f.name)));

      const addBtn = h("button.link-btn.pod-add", {
        type: "button",
        onclick: () => {
          const url = prompt("Podcast RSS feed URL:");
          if (!url?.trim()) return;
          const name = prompt("Show name:", "Podcast") || "Podcast";
          store.update((s) => {
            if (!s.podcasts) s.podcasts = { feeds: [] };
            s.podcasts.feeds = [...feeds(), { name: name.trim().slice(0, 30), url: url.trim() }];
            s.podcasts.active = url.trim();
          }, "podcasts");
          active = url.trim();
          draw();
        },
      }, "+ add show");

      clear(body).append(h("div.pod-head", {}, tabs, addBtn));
      if (!feeds().length) {
        body.append(h("div.muted.small.pod-empty", {}, "No shows yet. Add a podcast RSS feed to start."));
        return;
      }
      const list = h("div.pod-list", {}, h("div.widget-loading", {}, "LOADING EPISODES…"));
      body.append(player, list);

      let data;
      try {
        data = await ctx.api.podcast(active);
      } catch (err) {
        clear(list).append(h("div.widget-error", {}, `Feed unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      clear(list);
      for (const ep of data.episodes.slice(0, 20)) {
        list.append(h("button.pod-ep", {
          type: "button", title: ep.audio ? "Play" : "No audio (offline sample)",
          onclick: () => {
            if (!ep.audio) { toast("No audio available offline"); return; }
            player.src = ep.audio; player.play().catch(() => {});
          },
        },
          h("span.pod-play", { "aria-hidden": "true" }, "▶"),
          h("div.pod-ep-main", {},
            h("div.pod-ep-title", {}, ep.title),
            h("div.muted.small", {}, [timeAgo(ep.published), ep.duration].filter(Boolean).join(" · "))),
        ));
      }
    };

    ctx.onRefresh(draw);
    draw();
  },
};
