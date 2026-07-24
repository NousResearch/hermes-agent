// SA MedBot — a clinical decision-support consult grounded in South African
// guidelines (STGs/EML, SAMF, HPCSA, SAHPRA). Streams from the medchat endpoint,
// which runs a dedicated medical persona (no dashboard tools). Decision support
// for a qualified clinician — not a diagnosis or a substitute for judgement.

import { h, clear } from "../utils.js";
import { openViewer } from "../viewer.js";

// A single live handler so other widgets (e.g. Drug Reference) can hand the
// MedBot a question via `window.dispatchEvent(new CustomEvent("hub:medbot-ask",
// { detail: { text } }))`. Re-registered per mount, so no listener leak.
let askHandler = null;

export default {
  type: "medbot",
  title: "SA MedBot",
  icon: "⚕️",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;
    const history = () => store.state.medbot?.history || [];
    let busy = false;

    const persist = (msgs) => store.update((s) => { s.medbot = { history: msgs.slice(-40) }; }, "medbot");

    // Run a query end-to-end (used by the form and by external asks).
    async function runQuery(text) {
      text = (text || "").trim();
      if (!text || busy) return;
      // Drive the Anatomy Explorer: highlight where a mentioned condition sits.
      window.dispatchEvent(new CustomEvent("hub:anatomy-highlight", { detail: { text } }));
      const msgs = [...history(), { role: "user", content: text }];
      busy = true;
      persist([...msgs, { role: "assistant", content: "" }]);
      draw();
      let acc = "";
      let result = null;
      try {
        result = await ctx.api.medChatStream(
          msgs.map((m) => ({ role: m.role, content: m.content })),
          (delta) => {
            acc += delta;
            const last = body.querySelector(".med-log .med-msg:last-child .med-text");
            if (last) last.textContent = acc;
            const l = body.querySelector(".med-log");
            if (l) l.scrollTop = l.scrollHeight;
          });
      } catch (err) {
        acc = acc || `Error: ${err.message}`;
      }
      busy = false;
      persist([...msgs, { role: "assistant", content: acc, sources: result?.sources || [] }]);
      draw();
    }

    const draw = () => {
      const log = h("div.med-log");
      if (!history().length) {
        log.append(h("div.med-intro.muted.small", {},
          "Ask a clinical question. Answers are grounded in South African guidelines "
          + "(STGs/EML, SAMF, HPCSA) and are decision support only — they don't replace "
          + "your clinical judgement, examination, or current official guidance."));
      }
      for (const m of history()) {
        const msg = h(`div.med-msg.med-${m.role}`, {},
          h("span.med-role", {}, m.role === "user" ? "YOU" : "MEDBOT"),
          h("div.med-text", {}, m.content));
        if (m.sources?.length) {
          msg.append(h("div.med-sources", {},
            h("span.med-sources-label.muted.small", {}, "SOURCES (PubMed)"),
            ...m.sources.map((s) => h("button.med-source", {
              type: "button", title: `${s.journal} · ${s.date}`,
              onclick: () => openViewer({ url: s.url, title: s.title, source: s.journal, mode: "embed" }),
            }, `[${s.pmid}] ${s.title}`))));
        }
        log.append(msg);
      }

      const input = h("textarea.input.med-input", {
        rows: 2, placeholder: "e.g. First-line management of newly diagnosed HIV-TB co-infection?",
        "aria-label": "Ask SA MedBot",
        onkeydown: (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); } },
      });
      const form = h("form.med-form", { onsubmit: (ev) => { ev.preventDefault(); send(); } },
        input,
        h("button.btn.btn-primary", { type: "submit", disabled: busy }, busy ? "…" : "Ask"));

      clear(body).append(log, form,
        h("div.muted.small.med-disclaimer", {}, "Decision support · verify against current SA STGs/EML"));
      log.scrollTop = log.scrollHeight;
      if (!busy) input.focus();

      function send() { const text = input.value.trim(); if (text) runQuery(text); }
    };

    draw();
    ctx.onStore((topic) => { if (topic === "replace") draw(); });

    // Accept questions handed over from other widgets (Drug Reference, etc.).
    if (askHandler) window.removeEventListener("hub:medbot-ask", askHandler);
    askHandler = (ev) => runQuery(ev.detail?.text);
    window.addEventListener("hub:medbot-ask", askHandler);
  },
};
