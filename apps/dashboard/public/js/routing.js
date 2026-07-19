// Model routing overrides (Jarvis Phase 1 UI). Shows the FAST / CORE / DEEP
// tiers and lets you override the model per tier. Precedence is env var >
// this file override > built-in default, so a tier pinned by an env var is
// shown locked. Overrides persist server-side in data/routing.json.

import { h, clear, toast } from "./utils.js";
import { api } from "./api.js";

let panelEl = null;

function closePanel() {
  panelEl?.remove();
  panelEl = null;
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && panelEl) closePanel();
});

const TIER_LABEL = {
  fast: "FAST — cheap, high-volume (summaries)",
  core: "CORE — default conversation + tools",
  deep: "DEEP — hard reasoning + escalation",
};

export async function openRouting() {
  closePanel();
  const body = h("div.sum-body.sources-body", {}, h("div.widget-loading", {}, "LOADING ROUTING…"));
  panelEl = h("div.sum-backdrop", {
    onclick: (ev) => { if (ev.target === panelEl) closePanel(); },
  },
    h("div.sum-pop.sources-pop", { role: "dialog", "aria-label": "Model routing" },
      h("header.sum-head", {},
        h("span.sum-title", {}, "MODEL ROUTING"),
        h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: closePanel }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(panelEl);

  async function draw() {
    let snap;
    try {
      snap = await api.routing();
    } catch (err) {
      clear(body).append(h("div.widget-error", {}, `Cannot load routing: ${err.message}`));
      return;
    }
    clear(body);
    body.append(h("p.muted.small", {},
      "Pick the model for each tier. Env vars (HERMES_HUB_MODEL_*) win over these and show as locked. Blank a field to fall back to the default."));

    const inputs = {};
    for (const tier of ["fast", "core", "deep"]) {
      const locked = snap.env_locked?.[tier];
      const input = h("input.input.routing-input", {
        type: "text",
        value: snap.overrides?.[tier] || "",
        placeholder: snap.defaults?.[tier] || "",
        disabled: locked,
        "aria-label": `${tier} tier model`,
      });
      inputs[tier] = input;
      body.append(h("div.routing-row", {},
        h("div.routing-meta", {},
          h("div.routing-tier", {}, TIER_LABEL[tier]),
          h("div.muted.small", {}, locked
            ? `Locked by env → ${snap.tiers[tier]}`
            : `Active: ${snap.tiers[tier]} · default: ${snap.defaults?.[tier]}`),
        ),
        input,
      ));
    }

    body.append(h("div.routing-actions", {},
      h("button.btn.btn-primary", {
        type: "button",
        onclick: async () => {
          const overrides = {};
          for (const tier of ["fast", "core", "deep"]) {
            if (!inputs[tier].disabled) overrides[tier] = inputs[tier].value.trim();
          }
          try {
            await api.setRouting(overrides);
            toast("Routing saved");
            draw();
          } catch (err) { toast(err.message, "error"); }
        },
      }, "Save"),
      h("button.btn", {
        type: "button",
        onclick: async () => {
          try {
            await api.setRouting({ fast: "", core: "", deep: "" });
            toast("Routing reset to defaults");
            draw();
          } catch (err) { toast(err.message, "error"); }
        },
      }, "Reset to defaults"),
    ));
  }

  await draw();
}
