// The "∑ AI" affordance: every piece of dashboard data gets a summarize
// button. Opens a small overlay with the assistant's summary (Claude when
// configured, local extractive summarizer otherwise).

import { h, clear } from "./utils.js";
import { api } from "./api.js";

let popEl = null;

function closeSummary() {
  popEl?.remove();
  popEl = null;
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && popEl) closeSummary();
});

export async function showSummary({ kind, title, content }) {
  closeSummary();
  const body = h("div.sum-body", {}, h("div.widget-loading", {}, "SUMMARIZING…"));
  const modeChip = h("span.sum-mode", {}, "…");
  popEl = h("div.sum-backdrop", {
    onclick: (ev) => { if (ev.target === popEl) closeSummary(); },
  },
    h("div.sum-pop", { role: "dialog", "aria-label": "AI summary" },
      h("header.sum-head", {},
        h("span.sum-title", {}, "∑ SUMMARY", title ? ` — ${title.slice(0, 60)}` : ""),
        modeChip,
        h("button.icon-btn", { type: "button", "aria-label": "Close summary", onclick: closeSummary }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(popEl);
  try {
    const res = await api.summarize(kind, title, content);
    modeChip.textContent = res.mode === "claude" ? "CLAUDE" : "LOCAL";
    clear(body);
    for (const line of res.summary.split("\n")) {
      body.append(h("p.sum-line", {}, line));
    }
  } catch (err) {
    modeChip.textContent = "ERROR";
    clear(body).append(h("div.widget-error", {}, `Summary failed: ${err.message}`));
  }
}

/**
 * Build a summarize button. `getPayload` is called lazily on click and must
 * return {kind, title, content} (content = the raw text to summarize).
 */
export function summarizeButton(getPayload, { label = "∑", cls = "icon-btn sum-btn", tip = "AI summary" } = {}) {
  return h(`button.${cls.split(" ").join(".")}`, {
    type: "button",
    title: tip,
    "aria-label": tip,
    onclick: (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const payload = getPayload();
      if (payload?.content) showSummary(payload);
    },
  }, label);
}
