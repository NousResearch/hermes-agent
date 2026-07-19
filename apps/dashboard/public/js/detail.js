// Generic "expand to a big window" overlay. Any widget that exports a
// `detail(body, ctx)` renderer gets a ⤢ button (added in main.js); clicking it
// opens that renderer in a large modal with more room for charts, tables and
// controls. One overlay at a time; Esc or backdrop closes it. Modeled on the
// viewer overlay so it shares the look and accessibility behavior.

import { h, clear } from "./utils.js";

let overlay = null;

export function closeDetail() {
  overlay?.remove();
  overlay = null;
  document.body.classList.remove("detail-open");
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && overlay) closeDetail();
});

/**
 * @param spec  the widget module (needs .title, .icon, .detail)
 * @param ctx   the widget ctx (api/store/setBadge…) so detail shares plumbing
 */
export function openDetail(spec, ctx) {
  if (typeof spec.detail !== "function") return;
  closeDetail();
  const body = h("div.detail-body");
  overlay = h("div.detail-backdrop", {
    onclick: (ev) => { if (ev.target === overlay) closeDetail(); },
  },
    h("div.detail-pop", { role: "dialog", "aria-modal": "true", "aria-label": `${spec.title} — detail` },
      h("header.detail-head", {},
        h("span.detail-mark", { "aria-hidden": "true" }, spec.icon || "◆"),
        h("span.detail-title", {}, (spec.title || "").toUpperCase()),
        h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: closeDetail }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(overlay);
  document.body.classList.add("detail-open");
  try {
    spec.detail(body, ctx);
  } catch (err) {
    clear(body).append(h("div.widget-error", {}, `Detail view failed: ${err.message}`));
  }
}
