// Learning Tracker — track progress through Claude/AI courses & skills. Pure
// client-side (persisted in state). Seeded with the Anthropic Academy tracks
// plus key docs; users can add their own and cycle status not-started → in
// progress → done. A progress bar summarises completion.

import { h, clear } from "../utils.js";
import { viewerLink } from "../viewer.js";

const SEED = [
  ["Anthropic Academy: Claude with the API", "https://anthropic.skilljar.com/"],
  ["Anthropic Academy: Claude Code in Action", "https://anthropic.skilljar.com/"],
  ["Anthropic Academy: MCP", "https://anthropic.skilljar.com/"],
  ["Prompt engineering overview", "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview"],
  ["Building effective agents", "https://www.anthropic.com/engineering/building-effective-agents"],
  ["Claude Code: full docs", "https://docs.claude.com/en/docs/claude-code/overview"],
  ["Anthropic Cookbook", "https://github.com/anthropics/anthropic-cookbook"],
];

const STATUS = ["todo", "doing", "done"];
const STATUS_LABEL = { todo: "○ To do", doing: "◐ In progress", done: "● Done" };

export default {
  type: "tracker",
  title: "Learning Tracker",
  icon: "🎓",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const items = () => store.state.tracker?.items || [];

    const ensureSeed = () => {
      if (!store.state.tracker) {
        store.update((s) => {
          s.tracker = { items: SEED.map(([title, url]) => ({ title, url, status: "todo" })) };
        }, "tracker");
      }
    };

    const setStatus = (idx, status) =>
      store.update((s) => { s.tracker.items[idx].status = status; }, "tracker");
    const addItem = (title) =>
      store.update((s) => { s.tracker.items.push({ title, url: "", status: "todo" }); }, "tracker");
    const removeItem = (idx) =>
      store.update((s) => { s.tracker.items.splice(idx, 1); }, "tracker");

    const draw = () => {
      ensureSeed();
      const list = items();
      const done = list.filter((i) => i.status === "done").length;
      const pct = list.length ? Math.round((done / list.length) * 100) : 0;

      const bar = h("div.trk-bar", { title: `${done}/${list.length} complete` },
        h("div.trk-fill", { style: `width:${pct}%` }));
      const head = h("div.trk-head", {},
        h("span.trk-count", {}, `${done}/${list.length} complete`),
        h("span.trk-pct", {}, `${pct}%`));

      const rows = list.map((it, idx) => {
        const titleEl = it.url
          ? viewerLink(h("a.trk-title", { href: it.url, target: "_blank", rel: "noopener noreferrer" }, it.title),
            { url: it.url, title: it.title, source: "Learn", mode: "embed" })
          : h("span.trk-title", {}, it.title);
        const cycle = h("button.btn.btn-tiny.trk-status", { class: `btn btn-tiny trk-status st-${it.status}`, type: "button" },
          STATUS_LABEL[it.status]);
        cycle.addEventListener("click", () => {
          const next = STATUS[(STATUS.indexOf(it.status) + 1) % STATUS.length];
          setStatus(idx, next); draw();
        });
        const del = h("button.icon-btn.trk-del", { type: "button", title: "Remove", "aria-label": `Remove ${it.title}` }, "✕");
        del.addEventListener("click", () => { removeItem(idx); draw(); });
        return h("div.trk-row", { class: `trk-row st-${it.status}` }, titleEl, cycle, del);
      });

      const input = h("input.input.trk-input", { type: "text", placeholder: "Add a course or skill…", "aria-label": "Add learning item" });
      const onAdd = () => { const v = input.value.trim(); if (v) { addItem(v); draw(); } };
      input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") onAdd(); });
      const addBtn = h("button.btn.btn-tiny", { type: "button" }, "Add");
      addBtn.addEventListener("click", onAdd);

      clear(body).append(head, bar, h("div.trk-list", {}, rows),
        h("div.trk-add", {}, input, addBtn));
    };

    ctx.onRefresh(draw);
    ctx.onStore((topic) => { if (topic === "replace") draw(); });
    draw();
  },
};
