// Clinical calculators + reference ranges, tuned for South African practice
// (SI units: creatinine µmol/L, glucose/calcium mmol/L). Pure client-side — no
// network, works offline. Decision support for a clinician; verify against
// current guidance. Formulas are standard and cited in-line.

import { h, clear } from "../utils.js";

const num = (v) => { const n = parseFloat(v); return Number.isFinite(n) ? n : null; };
const round = (n, d = 1) => (n == null ? null : Math.round(n * 10 ** d) / 10 ** d);
const UMOL_TO_MGDL = 1 / 88.4;

// Each calculator: inputs [{key,label,unit,type,options}] and compute(vals)->
// {value, unit, tone, note}. tone ∈ good|warn|bad|info for the result colour.
const CALCULATORS = [
  {
    id: "egfr", name: "eGFR (CKD-EPI 2021)",
    blurb: "Race-free CKD-EPI 2021 creatinine equation.",
    inputs: [
      { key: "age", label: "Age", unit: "years", type: "number" },
      { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
      { key: "scr", label: "Creatinine", unit: "µmol/L", type: "number" },
    ],
    compute(v) {
      const age = num(v.age), scr = num(v.scr);
      if (age == null || scr == null || !v.sex) return null;
      const female = v.sex === "female";
      const k = female ? 0.7 : 0.9;
      const a = female ? -0.241 : -0.302;
      const cr = scr * UMOL_TO_MGDL;               // mg/dL
      let e = 142 * Math.min(cr / k, 1) ** a * Math.max(cr / k, 1) ** -1.200
        * 0.9938 ** age * (female ? 1.012 : 1);
      e = round(e, 0);
      const stage = e >= 90 ? "G1 (normal/high)" : e >= 60 ? "G2 (mild ↓)"
        : e >= 45 ? "G3a (mild–mod ↓)" : e >= 30 ? "G3b (mod–severe ↓)"
          : e >= 15 ? "G4 (severe ↓)" : "G5 (kidney failure)";
      const tone = e >= 60 ? "good" : e >= 30 ? "warn" : "bad";
      return { value: e, unit: "mL/min/1.73m²", tone, note: `CKD stage ${stage}. Review nephrotoxic/renally-cleared drug doses.` };
    },
  },
  {
    id: "crcl", name: "Creatinine clearance (Cockcroft-Gault)",
    blurb: "For renal drug dosing. Uses actual body weight.",
    inputs: [
      { key: "age", label: "Age", unit: "years", type: "number" },
      { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
      { key: "scr", label: "Creatinine", unit: "µmol/L", type: "number" },
    ],
    compute(v) {
      const age = num(v.age), wt = num(v.wt), scr = num(v.scr);
      if (age == null || wt == null || scr == null || !v.sex) return null;
      const cr = scr * UMOL_TO_MGDL;
      let crcl = ((140 - age) * wt * (v.sex === "female" ? 0.85 : 1)) / (72 * cr);
      crcl = round(crcl, 0);
      const tone = crcl >= 60 ? "good" : crcl >= 30 ? "warn" : "bad";
      return { value: crcl, unit: "mL/min", tone, note: "Many drug labels dose by Cockcroft-Gault CrCl, not eGFR." };
    },
  },
  {
    id: "bmi", name: "Body mass index (BMI)",
    blurb: "WHO adult categories.",
    inputs: [
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
      { key: "ht", label: "Height", unit: "cm", type: "number" },
    ],
    compute(v) {
      const wt = num(v.wt), ht = num(v.ht);
      if (wt == null || ht == null || ht === 0) return null;
      const m = ht / 100;
      const bmi = round(wt / (m * m), 1);
      const cat = bmi < 18.5 ? "underweight" : bmi < 25 ? "normal"
        : bmi < 30 ? "overweight" : bmi < 35 ? "obese I" : bmi < 40 ? "obese II" : "obese III";
      const tone = bmi >= 18.5 && bmi < 25 ? "good" : bmi < 30 ? "warn" : "bad";
      return { value: bmi, unit: "kg/m²", tone, note: `WHO: ${cat}.` };
    },
  },
  {
    id: "cca", name: "Corrected calcium",
    blurb: "Albumin-adjusted (SI).",
    inputs: [
      { key: "ca", label: "Measured calcium", unit: "mmol/L", type: "number" },
      { key: "alb", label: "Albumin", unit: "g/L", type: "number" },
    ],
    compute(v) {
      const ca = num(v.ca), alb = num(v.alb);
      if (ca == null || alb == null) return null;
      const cca = round(ca + 0.02 * (40 - alb), 2);
      const tone = cca >= 2.15 && cca <= 2.55 ? "good" : "warn";
      return { value: cca, unit: "mmol/L", tone, note: "Ref 2.15–2.55 mmol/L. Adds 0.02 per g/L albumin below 40." };
    },
  },
  {
    id: "map", name: "Mean arterial pressure (MAP)",
    blurb: "DBP + (SBP−DBP)/3.",
    inputs: [
      { key: "sbp", label: "Systolic", unit: "mmHg", type: "number" },
      { key: "dbp", label: "Diastolic", unit: "mmHg", type: "number" },
    ],
    compute(v) {
      const s = num(v.sbp), d = num(v.dbp);
      if (s == null || d == null) return null;
      const map = round(d + (s - d) / 3, 0);
      const tone = map >= 65 ? "good" : "bad";
      return { value: map, unit: "mmHg", tone, note: "Aim ≥65 mmHg for organ perfusion." };
    },
  },
  {
    id: "peds", name: "Paediatric maintenance fluids",
    blurb: "Holliday-Segar (4-2-1).",
    inputs: [{ key: "wt", label: "Weight", unit: "kg", type: "number" }],
    compute(v) {
      const wt = num(v.wt);
      if (wt == null || wt <= 0) return null;
      let hr;
      if (wt <= 10) hr = wt * 4;
      else if (wt <= 20) hr = 40 + (wt - 10) * 2;
      else hr = 60 + (wt - 20) * 1;
      const day = round(hr * 24, 0);
      return { value: round(hr, 0), unit: "mL/hr", tone: "info", note: `≈ ${day} mL/24h. Reassess with losses/status.` };
    },
  },
  {
    id: "chadsvasc", name: "CHA₂DS₂-VASc (AF stroke risk)",
    blurb: "Non-valvular AF; guides anticoagulation.",
    inputs: [
      { key: "chf", label: "Congestive HF / LV dysfunction", type: "check" },
      { key: "htn", label: "Hypertension", type: "check" },
      { key: "age75", label: "Age ≥ 75 (2)", type: "check" },
      { key: "dm", label: "Diabetes", type: "check" },
      { key: "stroke", label: "Prior stroke / TIA / thromboembolism (2)", type: "check" },
      { key: "vasc", label: "Vascular disease", type: "check" },
      { key: "age65", label: "Age 65–74", type: "check" },
      { key: "female", label: "Female", type: "check" },
    ],
    compute(v) {
      const s = (v.chf ? 1 : 0) + (v.htn ? 1 : 0) + (v.age75 ? 2 : 0) + (v.dm ? 1 : 0)
        + (v.stroke ? 2 : 0) + (v.vasc ? 1 : 0) + (v.age65 ? 1 : 0) + (v.female ? 1 : 0);
      const risk = [0.2, 0.6, 2.2, 3.2, 4.8, 7.2, 9.7, 11.2, 10.8, 12.2][Math.min(s, 9)];
      const tone = s === 0 ? "good" : s === 1 ? "warn" : "bad";
      const rec = s === 0 ? "Generally no antithrombotic."
        : s === 1 ? "Consider oral anticoagulation (esp. if not solely female)."
          : "Oral anticoagulation recommended.";
      return { value: s, unit: "points", tone, note: `≈ ${risk}%/yr stroke. ${rec}` };
    },
  },
];

// SA/NHLS-style adult reference ranges (SI units). Verify against your lab.
const REFERENCE = [
  ["Sodium", "135–145", "mmol/L"], ["Potassium", "3.5–5.1", "mmol/L"],
  ["Urea", "2.1–7.1", "mmol/L"], ["Creatinine (M / F)", "64–104 / 49–90", "µmol/L"],
  ["Glucose (fasting)", "3.9–5.6", "mmol/L"], ["Calcium (corrected)", "2.15–2.55", "mmol/L"],
  ["Magnesium", "0.66–1.07", "mmol/L"], ["Phosphate", "0.78–1.42", "mmol/L"],
  ["Albumin", "35–52", "g/L"], ["Bilirubin (total)", "5–21", "µmol/L"],
  ["ALT", "10–40", "U/L"], ["CRP", "< 5", "mg/L"],
  ["Haemoglobin (M / F)", "13–17 / 12–15", "g/dL"], ["Platelets", "150–400", "×10⁹/L"],
  ["White cells", "4–11", "×10⁹/L"], ["INR (off warfarin)", "0.8–1.2", ""],
  ["TSH", "0.27–4.2", "mIU/L"], ["HbA1c (target)", "< 7 (individualise)", "%"],
];

export default {
  type: "calc",
  title: "Clinical Tools",
  icon: "🧮",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    let active = store.state.calc?.active || CALCULATORS[0].id;
    const values = store.state.calc?.values || {};

    const persist = () => store.update((s) => { s.calc = { active, values }; }, "calc");

    const draw = () => {
      const picker = h("select.select.calc-picker", {
        "aria-label": "Calculator",
        onchange: (ev) => { active = ev.target.value; persist(); draw(); },
      }, [
        ...CALCULATORS.map((c) => h("option", { value: c.id, selected: c.id === active }, c.name)),
        h("option", { value: "reference", selected: active === "reference" }, "Reference ranges"),
      ]);

      if (active === "reference") {
        clear(body).append(picker, h("div.calc-refs", {},
          h("table.calc-ref-table", {}, h("tbody", {}, REFERENCE.map(([name, range, unit]) =>
            h("tr", {},
              h("td.calc-ref-name", {}, name),
              h("td.calc-ref-range", {}, range),
              h("td.calc-ref-unit.muted", {}, unit)))))),
          h("div.muted.small.calc-note", {}, "Adult SI reference ranges (SA/NHLS-style) · always verify against your reporting laboratory."));
        return;
      }

      const calc = CALCULATORS.find((c) => c.id === active);
      const result = h("div.calc-result");
      const paint = () => {
        clear(result);
        const out = calc.compute(values);
        if (!out) { result.append(h("span.muted.small", {}, "Enter values above.")); return; }
        result.append(
          h("div.calc-out", { class: `calc-out calc-${out.tone}` },
            h("span.calc-val", {}, String(out.value)),
            h("span.calc-unit", {}, out.unit)),
          h("div.small.calc-interp", {}, out.note));
      };

      const fields = calc.inputs.map((inp) => {
        if (inp.type === "check") {
          const cb = h("input", {
            type: "checkbox", checked: !!values[inp.key],
            onchange: (ev) => { values[inp.key] = ev.target.checked; persist(); paint(); },
          });
          return h("label.calc-check", {}, cb, h("span", {}, inp.label));
        }
        if (inp.type === "select") {
          return h("label.calc-field", {},
            h("span.calc-label", {}, inp.label),
            h("select.select", {
              onchange: (ev) => { values[inp.key] = ev.target.value; persist(); paint(); },
            }, [h("option", { value: "" }, "—"), ...inp.options.map((o) =>
              h("option", { value: o, selected: values[inp.key] === o }, o))]));
        }
        return h("label.calc-field", {},
          h("span.calc-label", {}, inp.label, inp.unit ? h("span.muted", {}, ` (${inp.unit})`) : null),
          h("input.input", {
            type: "number", step: "any", inputmode: "decimal", value: values[inp.key] ?? "",
            oninput: (ev) => { values[inp.key] = ev.target.value; persist(); paint(); },
          }));
      });

      clear(body).append(picker,
        h("div.muted.small.calc-blurb", {}, calc.blurb),
        h("div.calc-inputs", { class: calc.inputs[0].type === "check" ? "calc-inputs calc-checks" : "calc-inputs" }, fields),
        result,
        h("div.muted.small.calc-note", {}, "Decision support · verify against current SA STGs/EML and your clinical judgement."));
      paint();
    };

    ctx.onRefresh(draw);
    draw();
  },
};
