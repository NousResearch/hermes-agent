// Self-evolution approval inbox (Jarvis Phase 6). Lists the agent's proposals
// for improving itself; you Apply or Dismiss each. Memory cleanup auto-applies
// (shown as already done); prompt/behaviour changes wait here for your click.
// Every apply snapshots the hub first, so it's one-click reversible.

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

const STATUS_LABEL = {
  pending: "PENDING", applied: "APPLIED", "auto-applied": "AUTO-APPLIED",
  dismissed: "DISMISSED", "rolled-back": "ROLLED BACK",
};

export async function openEvolve() {
  closePanel();
  const body = h("div.sum-body.sources-body", {}, h("div.widget-loading", {}, "LOADING PROPOSALS…"));
  panelEl = h("div.sum-backdrop", {
    onclick: (ev) => { if (ev.target === panelEl) closePanel(); },
  },
    h("div.sum-pop.sources-pop", { role: "dialog", "aria-label": "Agent proposals" },
      h("header.sum-head", {},
        h("span.sum-title", {}, "AGENT PROPOSALS"),
        h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: closePanel }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(panelEl);

  async function draw() {
    let data;
    try {
      data = await api.evolve();
    } catch (err) {
      clear(body).append(h("div.widget-error", {}, `Cannot load proposals: ${err.message}`));
      return;
    }
    clear(body);

    body.append(h("div.evolve-head", {},
      h("p.muted.small", {}, "The agent reviews its own telemetry and memory and proposes small, reversible improvements. Memory cleanup applies automatically; everything else waits for you."),
      h("button.btn.btn-primary", {
        type: "button",
        onclick: async () => {
          try {
            const res = await api.evolveReflect();
            toast(res.created.length ? `${res.created.length} new proposal(s)` : "Nothing to change");
            draw();
          } catch (err) { toast(err.message, "error"); }
        },
      }, "Reflect now")));

    const open = data.proposals.filter((p) => p.status === "pending");
    const done = data.proposals.filter((p) => p.status !== "pending").slice(-8).reverse();

    if (!open.length) body.append(h("p.muted.small.sources-empty", {}, "No proposals awaiting review."));
    for (const p of open) body.append(proposalRow(p, draw, true));
    if (done.length) {
      body.append(h("div.muted.small.evolve-section", {}, "HISTORY"));
      for (const p of done) body.append(proposalRow(p, draw, false));
    }
  }

  await draw();
}

function proposalRow(p, redraw, actionable) {
  const controls = actionable
    ? h("div.evolve-actions", {},
      h("button.btn.btn-primary", {
        type: "button",
        onclick: async () => {
          try { await api.evolveProposal("apply", p.id); toast("Applied (snapshot saved)"); redraw(); }
          catch (err) { toast(err.message, "error"); }
        },
      }, "Apply"),
      h("button.btn", {
        type: "button",
        onclick: async () => {
          try { await api.evolveProposal("dismiss", p.id); toast("Dismissed"); redraw(); }
          catch (err) { toast(err.message, "error"); }
        },
      }, "Dismiss"))
    : h("div.evolve-history-actions", {},
      h("span.evolve-status", { class: `evolve-status evolve-${p.status}` }, STATUS_LABEL[p.status] || p.status),
      // Applied changes that captured a snapshot can be reverted in one click.
      (p.status === "applied" || p.status === "auto-applied") && p.snapshot
        ? h("button.btn.evolve-rollback", {
            type: "button",
            title: "Restore the snapshot taken before this change",
            onclick: async () => {
              try { await api.evolveProposal("rollback", p.id); toast("Rolled back (snapshot restored)"); redraw(); }
              catch (err) { toast(err.message, "error"); }
            },
          }, "Roll back")
        : null);

  return h("div.evolve-row", {},
    h("div.evolve-main", {},
      h("div.evolve-title", {}, p.title),
      h("div.muted.small", {}, p.rationale),
      p.payload && p.payload.text ? h("div.evolve-payload.small", {}, `“${p.payload.text}”`) : null),
    controls);
}
