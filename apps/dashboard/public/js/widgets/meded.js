// Medical education & OSCE practice, South African context. Curated station
// stems and high-yield study cards work offline; the "Practice"/"Teach"/"Quiz"
// actions hand a structured prompt to the SA MedBot (same page) for interactive,
// SA-grounded practice. Study aid — verify against current STGs/EML.

import { h, clear, toast } from "../utils.js";

const askMedbot = (text) => {
  window.dispatchEvent(new CustomEvent("hub:medbot-ask", { detail: { text } }));
  toast("Sent to SA MedBot ↑");
};

// OSCE stations grouped by domain (common SA finals / HPCSA-style).
const STATIONS = [
  ["History", "Newly diagnosed HIV — counsel & initiate ART", "TLD, adherence, VL monitoring, disclosure, TPT."],
  ["History", "Chronic cough — TB history & risk", "Xpert, HIV status, contacts, constitutional symptoms."],
  ["History", "Chest pain — assess for ACS", "Risk factors, character, timing, red flags, ECG plan."],
  ["History", "Headache — identify red flags", "Thunderclap, meningism, focal signs, raised ICP."],
  ["Examination", "Diabetic foot examination", "Neuropathy, pulses, deformity, ulcers, footwear advice."],
  ["Examination", "Respiratory examination", "Effusion vs consolidation vs TB signs; present findings."],
  ["Examination", "Cardiovascular examination", "Murmurs, heart failure signs, JVP, oedema."],
  ["Examination", "Thyroid examination", "Goitre, nodule, bruit, eye/status signs."],
  ["Practical", "Demonstrate MDI + spacer technique", "Steps, common errors, adherence check."],
  ["Practical", "Counsel on insulin injection technique", "Sites, rotation, storage, hypo recognition."],
  ["Data", "Interpret a CXR — pulmonary TB", "Zonal predilection, cavitation, effusion, miliary."],
  ["Data", "Interpret an ECG — STEMI / hyperkalaemia", "Territory, reciprocal changes, K⁺ changes."],
  ["Data", "Interpret an ABG", "Systematic acid-base, compensation, A–a gradient."],
  ["Data", "Interpret U&E — AKI / hyperkalaemia", "Pattern, urgency, management steps."],
  ["Communication", "Break bad news — new cancer diagnosis", "SPIKES, empathy, next steps, support."],
  ["Communication", "Counsel on TB treatment adherence", "Regimen, duration, side-effects, DOT, defaulting."],
  ["Communication", "PMTCT counselling", "Maternal ART, infant prophylaxis, feeding, testing."],
  ["Communication", "Obtain informed consent", "Nature, risks, alternatives, capacity, voluntariness."],
  ["Emergency", "Manage anaphylaxis", "IM adrenaline, airway, fluids, adjuncts, observation."],
  ["Emergency", "Manage diabetic ketoacidosis", "Fluids, fixed-rate insulin, K⁺, precipitant, monitoring."],
  ["Emergency", "Manage eclampsia / severe pre-eclampsia", "MgSO₄, BP control, delivery plan, monitoring."],
  ["Emergency", "Approach to the collapsed patient", "ABCDE, resuscitation, differentials, escalation."],
];

// High-yield SA study cards (concise, guideline-level; verify against STGs/EML).
const STUDY = [
  ["HIV & ART", ["First-line: fixed-dose TLD (TDF + 3TC + DTG).",
    "Test-and-treat; baseline CrAg/TB screen where indicated.",
    "VL at 6 & 12 months then annually; U=U.",
    "Offer TPT and cotrimoxazole per criteria."]],
  ["Tuberculosis", ["Xpert MTB/RIF Ultra is first-line diagnostic.",
    "DS-TB: 2 months RHZE then 4 months RH (fixed-dose, weight-banded).",
    "Screen every TB patient for HIV and vice versa.",
    "DR-TB: NDoH bedaquiline-based regimens + specialist input."]],
  ["Hypertension", ["Confirm with repeat/ambulatory readings.",
    "Lifestyle + EML agents by level of care; assess end-organ damage.",
    "Look for secondary causes if young/resistant.",
    "Target individualised; check adherence before escalating."]],
  ["Type-2 diabetes", ["Diagnose on HbA1c / fasting / OGTT.",
    "Metformin first-line unless contraindicated (eGFR).",
    "Annual foot, eye, renal (ACR) screening.",
    "Address CV risk: BP, lipids, smoking."]],
  ["Pre-eclampsia", ["BP ≥140/90 + proteinuria/end-organ features after 20 weeks.",
    "MgSO₄ for severe features / eclampsia prophylaxis.",
    "Control BP; plan timing of delivery (definitive treatment).",
    "Monitor for HELLP, pulmonary oedema, AKI."]],
  ["Paediatric diarrhoea (IMCI)", ["Assess dehydration; classify (none/some/severe).",
    "ORS + zinc; IV fluids if severe/shock.",
    "Continue feeding/breastfeeding; danger signs.",
    "Antibiotics only for dysentery/cholera per guidance."]],
  ["Sepsis", ["Recognise early (qSOFA/SIRS + suspected infection).",
    "Cultures then early broad-spectrum EML antibiotics.",
    "Fluids, source control, lactate, reassess.",
    "De-escalate on cultures (stewardship)."]],
  ["Anaphylaxis", ["IM adrenaline 1:1000 (adult 0.5 mg) — first and repeatable.",
    "Airway, high-flow O₂, IV fluids, position.",
    "Adjuncts (antihistamine/steroid) are secondary.",
    "Observe for biphasic reaction; adrenaline auto-injector + referral."]],
];

const DOMAINS = ["all", ...new Set(STATIONS.map((s) => s[0]))];

export default {
  type: "meded",
  title: "Med Ed & OSCE",
  icon: "🎓",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;
    let mode = store.state.meded?.mode || "osce";
    let domain = store.state.meded?.domain || "all";
    let topic = store.state.meded?.topic || "";
    const persist = () => store.update((s) => { s.meded = { mode, domain, topic }; }, "meded");

    const modeTabs = () => h("div.tabs.meded-modes", { role: "tablist" },
      [["osce", "OSCE"], ["study", "Study"], ["ask", "Ask"]].map(([m, label]) =>
        h("button.tab", {
          type: "button", role: "tab", "aria-selected": String(m === mode),
          onclick: () => { mode = m; persist(); draw(); },
        }, label)));

    const osceView = () => {
      const chips = h("div.meded-domains", {}, DOMAINS.map((d) =>
        h("button.meded-chip", {
          type: "button", class: d === domain ? "meded-chip meded-chip-on" : "meded-chip",
          onclick: () => { domain = d; persist(); draw(); },
        }, d === "all" ? "All" : d)));
      const list = h("div.meded-stations");
      const shown = STATIONS.filter((s) => domain === "all" || s[0] === domain);
      for (const [dom, title, brief] of shown) {
        list.append(h("div.meded-station", {},
          h("div.meded-station-main", {},
            h("div.meded-station-title", {}, title),
            h("div.muted.small.meded-station-brief", {}, h("span.meded-dom", {}, dom), " · ", brief)),
          h("div.meded-station-actions", {},
            h("button.btn.btn-primary.btn-tiny", {
              type: "button", title: "Interactive OSCE with the SA MedBot",
              onclick: () => askMedbot(
                `Run an interactive South African OSCE station as the examiner. Station: "${title}" (${dom}). `
                + `Give me the candidate instructions and the clinical scenario, then pause for my approach. `
                + `Mark against a structured SA scheme aligned to the STGs/EML and HPCSA, and give feedback at the end. Begin.`),
            }, "Practice"),
            h("button.btn.btn-tiny", {
              type: "button", title: "Model answer / marking points",
              onclick: () => askMedbot(
                `Give a concise structured model answer for this South African OSCE station: "${title}" (${dom}) — `
                + `the ideal candidate approach and the key marking points, per SA guidelines.`),
            }, "Model"))));
      }
      const random = h("button.link-btn.meded-random", {
        type: "button",
        onclick: () => { const s = shown[Math.floor(Math.random() * shown.length)];
          if (s) askMedbot(`Run an interactive South African OSCE station as the examiner: "${s[1]}" (${s[0]}). Begin with the scenario, then pause for my approach; mark me at the end.`); },
      }, "🎲 Random station");
      clear(body).append(modeTabs(), chips, random, list);
    };

    const studyView = () => {
      const cards = h("div.meded-study");
      for (const [t, points] of STUDY) {
        cards.append(h("div.meded-card", {},
          h("div.meded-card-head", {},
            h("span.meded-card-title", {}, t),
            h("button.link-btn.btn-tiny", {
              type: "button",
              onclick: () => askMedbot(`Teach me, in South African context (STGs/EML, SAMF), the high-yield points for ${t}. Structure it for exam revision.`),
            }, "Go deeper →")),
          h("ul.meded-points", {}, points.map((p) => h("li", {}, p)))));
      }
      clear(body).append(modeTabs(),
        h("div.muted.small.meded-note", {}, "High-yield revision cards · verify against the current SA STGs/EML."),
        cards);
    };

    const askView = () => {
      const input = h("input.input.meded-topic", {
        type: "text", value: topic, placeholder: "Topic, e.g. hyperkalaemia, PPH, meningitis…",
        "aria-label": "Study topic",
        oninput: (ev) => { topic = ev.target.value; persist(); },
      });
      const launch = (build, needTopic = true) => () => {
        const t = input.value.trim();
        if (needTopic && !t) { toast("Enter a topic first"); return; }
        askMedbot(build(t));
      };
      clear(body).append(modeTabs(),
        h("div.muted.small.meded-note", {}, "Hand a study prompt to the SA MedBot (needs an API key for live answers)."),
        input,
        h("div.meded-ask-actions", {},
          h("button.btn.btn-primary", { type: "button",
            onclick: launch((t) => `Teach me about ${t} in South African clinical context (STGs/EML, SAMF), structured for revision.`) }, "Teach me"),
          h("button.btn", { type: "button",
            onclick: launch((t) => `Give me 5 single-best-answer questions on ${t} in South African context, with answers and brief explanations. Ask them one at a time.`) }, "Quiz me"),
          h("button.btn", { type: "button",
            onclick: launch((t) => `Present an interactive South African clinical case on ${t}. Reveal it step by step and ask me to reason through assessment and management.`) }, "Case")));
    };

    const draw = () => {
      if (mode === "study") return studyView();
      if (mode === "ask") return askView();
      return osceView();
    };

    ctx.onRefresh(draw);
    draw();
  },
};
