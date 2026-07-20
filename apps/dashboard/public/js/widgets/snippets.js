// Prompt & snippet library — save, tag, search and copy your own reusable
// prompts, skills and code snippets. Local + synced state. Seeded with a few
// high-leverage Claude/coding prompts you can edit or delete.

import { h, clear, toast, uid } from "../utils.js";

const SEED = [
  { id: "s1", tag: "Prompt", title: "Explain like a senior engineer",
    body: "Explain <topic> as a senior engineer would to a mid-level dev: the mental model first, then the 20% that matters most, then one common pitfall. Be concise." },
  { id: "s2", tag: "Claude Code", title: "Plan before coding",
    body: "Before writing any code, outline your plan: the files you'll touch, the approach, and the trade-offs. Wait for my OK, then implement with tests." },
  { id: "s3", tag: "Prompt", title: "Rubber-duck a bug",
    body: "Here's a bug: <describe>. Ask me clarifying questions one at a time, form a hypothesis, and propose the smallest experiment to confirm it before suggesting a fix." },
  { id: "s4", tag: "Skill idea", title: "Review my diff",
    body: "Review the current git diff for correctness bugs and simplifications. Rank by severity; for each: the failure scenario, the file:line, and a concrete fix." },
  { id: "s5", tag: "Code", title: "Debounce (JS)",
    body: "const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };" },
];

export default {
  type: "snippets",
  title: "Prompt Library",
  icon: "📎",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const items = () => store.state.snippets?.items || SEED;
    let query = "";
    let editing = null; // id being edited, or "new"

    const persist = (list) => store.update((s) => { s.snippets = { items: list }; }, "snippets");

    const copy = (text) => {
      if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(() => toast("Copied"), () => toast("Copy failed", "error"));
      else toast("Clipboard unavailable");
    };

    const matches = (it) => {
      if (!query) return true;
      const hay = `${it.title} ${it.body} ${it.tag}`.toLowerCase();
      return query.split(/\s+/).every((t) => hay.includes(t));
    };

    const draw = () => {
      const search = h("input.input.snip-search", {
        type: "search", placeholder: "Search prompts & snippets…", value: query,
        "aria-label": "Search snippets",
        oninput: (ev) => { query = ev.target.value.trim().toLowerCase(); renderList(); },
      });
      const addBtn = h("button.btn.btn-tiny.snip-add", {
        type: "button", onclick: () => { editing = "new"; draw(); },
      }, "+ New");

      const list = h("div.snip-list");
      const renderList = () => {
        clear(list);
        const shown = items().filter(matches);
        if (!shown.length) { list.append(h("div.muted.small", {}, query ? "No matches." : "No snippets yet.")); return; }
        for (const it of shown) {
          list.append(h("div.snip-item", {},
            h("div.snip-item-head", {},
              h("span.snip-tag", {}, it.tag || "Note"),
              h("span.snip-title", {}, it.title),
              h("div.snip-item-actions", {},
                h("button.icon-btn.snip-btn", { type: "button", title: "Copy", onclick: () => copy(it.body) }, "⧉"),
                h("button.icon-btn.snip-btn", { type: "button", title: "Edit", onclick: () => { editing = it.id; draw(); } }, "✎"),
                h("button.icon-btn.snip-btn", { type: "button", title: "Delete",
                  onclick: () => { persist(items().filter((x) => x.id !== it.id)); draw(); } }, "✕"))),
            h("pre.snip-body", {}, it.body)));
        }
      };

      if (editing) {
        const cur = editing === "new" ? { tag: "Prompt", title: "", body: "" } : items().find((x) => x.id === editing) || {};
        const tag = h("input.input.snip-f-tag", { value: cur.tag || "", placeholder: "Tag (Prompt / Code / Skill idea…)" });
        const title = h("input.input.snip-f-title", { value: cur.title || "", placeholder: "Title" });
        const bodyEl = h("textarea.input.snip-f-body", { rows: 5, placeholder: "Prompt or snippet text…" }, cur.body || "");
        clear(body).append(
          h("div.snip-form", {},
            h("div.snip-form-row", {}, tag, title),
            bodyEl,
            h("div.snip-form-actions", {},
              h("button.btn.btn-primary", { type: "button", onclick: () => {
                const t = title.value.trim(); if (!t) { toast("Title required"); return; }
                const entry = { id: editing === "new" ? uid() : editing, tag: tag.value.trim() || "Note", title: t, body: bodyEl.value };
                const list2 = editing === "new" ? [entry, ...items()] : items().map((x) => x.id === editing ? entry : x);
                persist(list2); editing = null; draw();
              } }, "Save"),
              h("button.btn", { type: "button", onclick: () => { editing = null; draw(); } }, "Cancel"))));
        return;
      }

      clear(body).append(h("div.snip-head", {}, search, addBtn), list);
      renderList();
    };

    ctx.onRefresh(draw);
    draw();
  },
};
