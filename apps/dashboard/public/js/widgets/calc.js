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
  {
    id: "gcs", name: "Glasgow Coma Scale", blurb: "Eye + Verbal + Motor.",
    inputs: [
      { key: "e", label: "Eye opening", type: "select", options: [
        { value: 4, label: "4 — spontaneous" }, { value: 3, label: "3 — to voice" },
        { value: 2, label: "2 — to pain" }, { value: 1, label: "1 — none" }] },
      { key: "v", label: "Verbal", type: "select", options: [
        { value: 5, label: "5 — oriented" }, { value: 4, label: "4 — confused" },
        { value: 3, label: "3 — inappropriate words" }, { value: 2, label: "2 — sounds" },
        { value: 1, label: "1 — none" }] },
      { key: "m", label: "Motor", type: "select", options: [
        { value: 6, label: "6 — obeys" }, { value: 5, label: "5 — localises" },
        { value: 4, label: "4 — withdraws" }, { value: 3, label: "3 — flexion" },
        { value: 2, label: "2 — extension" }, { value: 1, label: "1 — none" }] },
    ],
    compute(v) {
      const e = num(v.e), vb = num(v.v), m = num(v.m);
      if (e == null || vb == null || m == null) return null;
      const s = e + vb + m;
      const sev = s >= 13 ? "minor" : s >= 9 ? "moderate" : "severe";
      const tone = s >= 13 ? "good" : s >= 9 ? "warn" : "bad";
      return { value: s, unit: "/15", tone, note: `${sev} (E${e}V${vb}M${m}). ≤8 → consider airway protection.` };
    },
  },
  {
    id: "curb65", name: "CURB-65 (pneumonia)", blurb: "CAP severity & disposition.",
    inputs: [
      { key: "conf", label: "Confusion (new)", type: "check" },
      { key: "urea", label: "Urea > 7 mmol/L", type: "check" },
      { key: "rr", label: "Respiratory rate ≥ 30", type: "check" },
      { key: "bp", label: "SBP < 90 or DBP ≤ 60", type: "check" },
      { key: "age", label: "Age ≥ 65", type: "check" },
    ],
    compute(v) {
      const s = (v.conf ? 1 : 0) + (v.urea ? 1 : 0) + (v.rr ? 1 : 0) + (v.bp ? 1 : 0) + (v.age ? 1 : 0);
      const disp = s <= 1 ? "low risk — consider outpatient" : s === 2 ? "admit (short/supervised)"
        : "severe — assess for ICU";
      const tone = s <= 1 ? "good" : s === 2 ? "warn" : "bad";
      return { value: s, unit: "/5", tone, note: disp };
    },
  },
  {
    id: "qsofa", name: "qSOFA (sepsis)", blurb: "Bedside sepsis risk outside ICU.",
    inputs: [
      { key: "rr", label: "Respiratory rate ≥ 22", type: "check" },
      { key: "ams", label: "Altered mentation (GCS < 15)", type: "check" },
      { key: "sbp", label: "SBP ≤ 100 mmHg", type: "check" },
    ],
    compute(v) {
      const s = (v.rr ? 1 : 0) + (v.ams ? 1 : 0) + (v.sbp ? 1 : 0);
      const tone = s >= 2 ? "bad" : "warn";
      return { value: s, unit: "/3", tone, note: s >= 2
        ? "≥2: higher risk — escalate assessment / sepsis workup."
        : "Lower risk; reassess if clinical concern." };
    },
  },
  {
    id: "wells", name: "Wells score (DVT)", blurb: "Pre-test probability of DVT.",
    inputs: [
      { key: "ca", label: "Active cancer", type: "check" },
      { key: "para", label: "Paralysis / immobilisation of leg", type: "check" },
      { key: "bed", label: "Bedridden >3d or surgery <12wk", type: "check" },
      { key: "tender", label: "Localised deep-vein tenderness", type: "check" },
      { key: "swell", label: "Entire leg swollen", type: "check" },
      { key: "calf", label: "Calf >3cm larger than other", type: "check" },
      { key: "oedema", label: "Pitting oedema (symptomatic leg)", type: "check" },
      { key: "veins", label: "Collateral superficial veins", type: "check" },
      { key: "prev", label: "Previous documented DVT", type: "check" },
      { key: "alt", label: "Alternative diagnosis ≥ as likely (−2)", type: "check" },
    ],
    compute(v) {
      let s = 0;
      ["ca", "para", "bed", "tender", "swell", "calf", "oedema", "veins", "prev"]
        .forEach((k) => { if (v[k]) s += 1; });
      if (v.alt) s -= 2;
      const cat = s >= 2 ? "DVT likely" : "DVT unlikely";
      const tone = s >= 2 ? "bad" : "good";
      return { value: s, unit: "points", tone, note: `${cat}. Combine with D-dimer / ultrasound per protocol.` };
    },
  },
  {
    id: "anion", name: "Anion gap", blurb: "Na − (Cl + HCO₃), albumin-corrected.",
    inputs: [
      { key: "na", label: "Sodium", unit: "mmol/L", type: "number" },
      { key: "cl", label: "Chloride", unit: "mmol/L", type: "number" },
      { key: "hco3", label: "Bicarbonate", unit: "mmol/L", type: "number" },
      { key: "alb", label: "Albumin (optional)", unit: "g/L", type: "number" },
    ],
    compute(v) {
      const na = num(v.na), cl = num(v.cl), hco3 = num(v.hco3);
      if (na == null || cl == null || hco3 == null) return null;
      const raw = round(na - (cl + hco3), 1);
      const alb = num(v.alb);
      const corr = alb != null ? round(raw + 0.25 * (40 - alb), 1) : null;
      const val = corr != null ? corr : raw;
      const tone = val > 12 ? "warn" : "good";
      return { value: val, unit: "mmol/L", tone,
        note: (corr != null ? `Albumin-corrected (raw ${raw}). ` : "")
          + "Ref ~8–12. High → HAGMA (lactate, ketones, renal, toxins)." };
    },
  },
  {
    id: "nacorr", name: "Corrected sodium (hyperglycaemia)", blurb: "+0.3 mmol/L Na per mmol/L glucose > 5.5.",
    inputs: [
      { key: "na", label: "Measured sodium", unit: "mmol/L", type: "number" },
      { key: "glu", label: "Glucose", unit: "mmol/L", type: "number" },
    ],
    compute(v) {
      const na = num(v.na), glu = num(v.glu);
      if (na == null || glu == null) return null;
      const c = round(na + 0.3 * (glu - 5.5), 1);
      const tone = c >= 135 && c <= 145 ? "good" : "warn";
      return { value: c, unit: "mmol/L", tone, note: "Corrected for glucose-driven dilution. Ref 135–145." };
    },
  },
  {
    id: "ibw", name: "Ideal body weight (Devine)", blurb: "For weight-based dosing.",
    inputs: [
      { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
      { key: "ht", label: "Height", unit: "cm", type: "number" },
    ],
    compute(v) {
      const ht = num(v.ht);
      if (ht == null || !v.sex) return null;
      const base = v.sex === "female" ? 45.5 : 50;
      const ibw = Math.max(base + 0.9 * (ht - 152), base);
      return { value: round(ibw, 1), unit: "kg", tone: "info",
        note: "Devine IBW. Use adjusted body weight if markedly obese." };
    },
  },
  {
    id: "pedsdose", name: "Paediatric dose (mg/kg)", blurb: "Weight-based dose → volume.",
    inputs: [
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
      { key: "dose", label: "Dose", unit: "mg/kg", type: "number" },
      { key: "conc", label: "Concentration (optional)", unit: "mg/mL", type: "number" },
    ],
    compute(v) {
      const wt = num(v.wt), dose = num(v.dose);
      if (wt == null || dose == null) return null;
      const mg = round(wt * dose, 1);
      const conc = num(v.conc);
      const ml = conc && conc > 0 ? round(mg / conc, 1) : null;
      return { value: mg, unit: "mg", tone: "info",
        note: (ml != null ? `≈ ${ml} mL. ` : "") + "Always check the maximum dose and formulation." };
    },
  },
  {
    id: "wellspe", name: "Wells score (PE)", blurb: "Pre-test probability of pulmonary embolism.",
    inputs: [
      { key: "dvt", label: "Clinical signs of DVT (3)", type: "check" },
      { key: "pe1", label: "PE is the most likely diagnosis (3)", type: "check" },
      { key: "hr", label: "Heart rate > 100 (1.5)", type: "check" },
      { key: "immob", label: "Immobilisation ≥3d / surgery <4wk (1.5)", type: "check" },
      { key: "prev", label: "Previous DVT / PE (1.5)", type: "check" },
      { key: "haem", label: "Haemoptysis (1)", type: "check" },
      { key: "malig", label: "Malignancy (1)", type: "check" },
    ],
    compute(v) {
      const s = (v.dvt ? 3 : 0) + (v.pe1 ? 3 : 0) + (v.hr ? 1.5 : 0) + (v.immob ? 1.5 : 0)
        + (v.prev ? 1.5 : 0) + (v.haem ? 1 : 0) + (v.malig ? 1 : 0);
      const cat = s > 4 ? "PE likely" : "PE unlikely";
      const tone = s > 4 ? "bad" : "good";
      return { value: round(s, 1), unit: "points", tone,
        note: `${cat} (>4 = likely). If unlikely, a negative D-dimer can exclude PE.` };
    },
  },
  {
    id: "hasbled", name: "HAS-BLED (bleeding risk)", blurb: "Bleeding risk on anticoagulation for AF.",
    inputs: [
      { key: "htn", label: "Uncontrolled hypertension (SBP >160)", type: "check" },
      { key: "renal", label: "Abnormal renal function", type: "check" },
      { key: "liver", label: "Abnormal liver function", type: "check" },
      { key: "stroke", label: "Prior stroke", type: "check" },
      { key: "bleed", label: "Prior major bleeding / predisposition", type: "check" },
      { key: "inr", label: "Labile INR", type: "check" },
      { key: "elderly", label: "Age > 65", type: "check" },
      { key: "drugs", label: "Drugs predisposing to bleeding", type: "check" },
      { key: "alcohol", label: "Alcohol ≥ 8 units/week", type: "check" },
    ],
    compute(v) {
      const s = ["htn", "renal", "liver", "stroke", "bleed", "inr", "elderly", "drugs", "alcohol"]
        .reduce((a, k) => a + (v[k] ? 1 : 0), 0);
      const tone = s >= 3 ? "bad" : s === 2 ? "warn" : "good";
      return { value: s, unit: "points", tone,
        note: s >= 3 ? "High bleeding risk — address modifiable factors; not a reason alone to withhold anticoagulation."
          : "Lower bleeding risk. Weigh against stroke risk (CHA₂DS₂-VASc)." };
    },
  },
  {
    id: "rcri", name: "RCRI (cardiac risk, non-cardiac surgery)", blurb: "Revised Cardiac Risk Index (Lee).",
    inputs: [
      { key: "surg", label: "High-risk surgery (intraperitoneal/thoracic/suprainguinal vascular)", type: "check" },
      { key: "ihd", label: "Ischaemic heart disease", type: "check" },
      { key: "chf", label: "Congestive heart failure", type: "check" },
      { key: "cva", label: "Cerebrovascular disease", type: "check" },
      { key: "dm", label: "Insulin-treated diabetes", type: "check" },
      { key: "cr", label: "Creatinine > 177 µmol/L", type: "check" },
    ],
    compute(v) {
      const s = ["surg", "ihd", "chf", "cva", "dm", "cr"].reduce((a, k) => a + (v[k] ? 1 : 0), 0);
      const risk = [0.4, 0.9, 6.6, 11][Math.min(s, 3)];
      const tone = s >= 2 ? "bad" : s === 1 ? "warn" : "good";
      return { value: s, unit: "points", tone, note: `≈ ${risk}% major cardiac event at 30 days.` };
    },
  },
  {
    id: "sirs", name: "SIRS criteria", blurb: "Systemic inflammatory response.",
    inputs: [
      { key: "temp", label: "Temp > 38 or < 36 °C", type: "check" },
      { key: "hr", label: "Heart rate > 90", type: "check" },
      { key: "rr", label: "RR > 20 or pCO₂ < 4.3 kPa", type: "check" },
      { key: "wcc", label: "WCC > 12 or < 4 ×10⁹/L", type: "check" },
    ],
    compute(v) {
      const s = ["temp", "hr", "rr", "wcc"].reduce((a, k) => a + (v[k] ? 1 : 0), 0);
      const tone = s >= 2 ? "warn" : "good";
      return { value: s, unit: "/4", tone,
        note: s >= 2 ? "SIRS present. With suspected infection, assess for sepsis (qSOFA/SOFA)." : "SIRS not met." };
    },
  },
  {
    id: "centor", name: "Centor / McIsaac (strep pharyngitis)", blurb: "Likelihood of group-A strep.",
    inputs: [
      { key: "exudate", label: "Tonsillar exudate/swelling", type: "check" },
      { key: "nodes", label: "Tender anterior cervical nodes", type: "check" },
      { key: "fever", label: "History of fever > 38 °C", type: "check" },
      { key: "cough", label: "Absence of cough", type: "check" },
      { key: "age", label: "Age", type: "select", options: [
        { value: 1, label: "3–14 (+1)" }, { value: 0, label: "15–44 (0)" }, { value: -1, label: "≥45 (−1)" }] },
    ],
    compute(v) {
      const age = num(v.age);
      if (age == null) return null;
      const s = (v.exudate ? 1 : 0) + (v.nodes ? 1 : 0) + (v.fever ? 1 : 0) + (v.cough ? 1 : 0) + age;
      const tone = s >= 4 ? "warn" : s >= 2 ? "info" : "good";
      const rec = s <= 0 ? "strep unlikely — no testing/antibiotics"
        : s <= 2 ? "low risk — testing optional" : "consider rapid test / empiric therapy per local guidance";
      return { value: s, unit: "points", tone, note: rec };
    },
  },
  {
    id: "childpugh", name: "Child-Pugh (cirrhosis)", blurb: "Severity/prognosis in chronic liver disease.",
    inputs: [
      { key: "bili", label: "Bilirubin", type: "select", options: [
        { value: 1, label: "< 34 µmol/L" }, { value: 2, label: "34–50" }, { value: 3, label: "> 50" }] },
      { key: "alb", label: "Albumin", type: "select", options: [
        { value: 1, label: "> 35 g/L" }, { value: 2, label: "28–35" }, { value: 3, label: "< 28" }] },
      { key: "inr", label: "INR", type: "select", options: [
        { value: 1, label: "< 1.7" }, { value: 2, label: "1.7–2.3" }, { value: 3, label: "> 2.3" }] },
      { key: "asc", label: "Ascites", type: "select", options: [
        { value: 1, label: "None" }, { value: 2, label: "Mild (controlled)" }, { value: 3, label: "Moderate–severe" }] },
      { key: "enc", label: "Encephalopathy", type: "select", options: [
        { value: 1, label: "None" }, { value: 2, label: "Grade 1–2" }, { value: 3, label: "Grade 3–4" }] },
    ],
    compute(v) {
      const parts = ["bili", "alb", "inr", "asc", "enc"].map((k) => num(v[k]));
      if (parts.some((p) => p == null)) return null;
      const s = parts.reduce((a, b) => a + b, 0);
      const cls = s <= 6 ? "A" : s <= 9 ? "B" : "C";
      const tone = cls === "A" ? "good" : cls === "B" ? "warn" : "bad";
      return { value: s, unit: `pts · class ${cls}`, tone,
        note: `Class ${cls} (A 5–6, B 7–9, C 10–15). Higher class → worse peri-operative and 1-yr survival.` };
    },
  },
  {
    id: "meldna", name: "MELD-Na", blurb: "Chronic liver disease severity / transplant priority.",
    inputs: [
      { key: "bili", label: "Bilirubin", unit: "µmol/L", type: "number" },
      { key: "inr", label: "INR", type: "number" },
      { key: "cr", label: "Creatinine", unit: "µmol/L", type: "number" },
      { key: "na", label: "Sodium", unit: "mmol/L", type: "number" },
      { key: "hd", label: "Dialysis ≥2× in last week", type: "check" },
    ],
    compute(v) {
      const bili = num(v.bili), inr = num(v.inr), cr = num(v.cr), na = num(v.na);
      if (bili == null || inr == null || cr == null || na == null) return null;
      const biliMg = Math.max(bili / 17.1, 1);
      let crMg = Math.max(cr / 88.4, 1);
      if (v.hd || crMg > 4) crMg = 4;
      const inrC = Math.max(inr, 1);
      let meld = Math.round(10 * (0.957 * Math.log(crMg) + 0.378 * Math.log(biliMg) + 1.120 * Math.log(inrC) + 0.643));
      meld = Math.max(6, Math.min(meld, 40));
      let score = meld;
      if (meld > 11) {
        const naC = Math.max(125, Math.min(na, 137));
        score = Math.round(meld + 1.32 * (137 - naC) - (0.033 * meld * (137 - naC)));
        score = Math.max(6, Math.min(score, 40));
      }
      const tone = score >= 30 ? "bad" : score >= 20 ? "warn" : "good";
      return { value: score, unit: "points", tone, note: `Higher = greater 90-day mortality (MELD ${meld}).` };
    },
  },
  {
    id: "qtc", name: "Corrected QT (Bazett)", blurb: "QTc = QT / √RR.",
    inputs: [
      { key: "qt", label: "QT interval", unit: "ms", type: "number" },
      { key: "hr", label: "Heart rate", unit: "bpm", type: "number" },
    ],
    compute(v) {
      const qt = num(v.qt), hr = num(v.hr);
      if (qt == null || hr == null || hr <= 0) return null;
      const rr = 60 / hr;
      const qtc = Math.round(qt / Math.sqrt(rr));
      const tone = qtc >= 500 ? "bad" : qtc >= 460 ? "warn" : "good";
      return { value: qtc, unit: "ms", tone,
        note: "Prolonged > 450 (M) / > 470 (F); > 500 → high torsades risk. Review QT-prolonging drugs & K⁺/Mg²⁺." };
    },
  },
  {
    id: "shock", name: "Shock index", blurb: "HR ÷ SBP.",
    inputs: [
      { key: "hr", label: "Heart rate", unit: "bpm", type: "number" },
      { key: "sbp", label: "Systolic BP", unit: "mmHg", type: "number" },
    ],
    compute(v) {
      const hr = num(v.hr), sbp = num(v.sbp);
      if (hr == null || sbp == null || sbp <= 0) return null;
      const si = round(hr / sbp, 2);
      const tone = si >= 1.0 ? "bad" : si >= 0.7 ? "warn" : "good";
      return { value: si, unit: "", tone, note: "Normal 0.5–0.7; > 0.9 suggests haemodynamic compromise." };
    },
  },
  {
    id: "bsa", name: "Body surface area (Mosteller)", blurb: "√(ht·wt / 3600).",
    inputs: [
      { key: "ht", label: "Height", unit: "cm", type: "number" },
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
    ],
    compute(v) {
      const ht = num(v.ht), wt = num(v.wt);
      if (ht == null || wt == null) return null;
      return { value: round(Math.sqrt((ht * wt) / 3600), 2), unit: "m²", tone: "info",
        note: "Used for chemotherapy and some drug dosing." };
    },
  },
  {
    id: "ldl", name: "LDL (Friedewald)", blurb: "TC − HDL − TG/2.2 (SI).",
    inputs: [
      { key: "tc", label: "Total cholesterol", unit: "mmol/L", type: "number" },
      { key: "hdl", label: "HDL", unit: "mmol/L", type: "number" },
      { key: "tg", label: "Triglycerides", unit: "mmol/L", type: "number" },
    ],
    compute(v) {
      const tc = num(v.tc), hdl = num(v.hdl), tg = num(v.tg);
      if (tc == null || hdl == null || tg == null) return null;
      if (tg >= 4.5) return { value: "—", unit: "", tone: "warn", note: "Invalid when TG ≥ 4.5 mmol/L — request a direct LDL." };
      return { value: round(tc - hdl - tg / 2.2, 2), unit: "mmol/L", tone: "info",
        note: "High-risk LDL target < 1.8 mmol/L." };
    },
  },
  {
    id: "apri", name: "APRI (liver fibrosis)", blurb: "AST-to-platelet ratio index.",
    inputs: [
      { key: "ast", label: "AST", unit: "U/L", type: "number" },
      { key: "uln", label: "AST upper limit (default 40)", unit: "U/L", type: "number" },
      { key: "plt", label: "Platelets", unit: "×10⁹/L", type: "number" },
    ],
    compute(v) {
      const ast = num(v.ast), plt = num(v.plt);
      if (ast == null || plt == null || plt <= 0) return null;
      const uln = num(v.uln) || 40;
      const apri = round(((ast / uln) / plt) * 100, 2);
      const tone = apri > 1 ? "warn" : "good";
      return { value: apri, unit: "", tone, note: "> 0.5 possible, > 1.0 significant fibrosis, > 2 cirrhosis (derived in viral hepatitis)." };
    },
  },
  {
    id: "fib4", name: "FIB-4 (liver fibrosis)", blurb: "(age·AST)/(platelets·√ALT).",
    inputs: [
      { key: "age", label: "Age", unit: "years", type: "number" },
      { key: "ast", label: "AST", unit: "U/L", type: "number" },
      { key: "alt", label: "ALT", unit: "U/L", type: "number" },
      { key: "plt", label: "Platelets", unit: "×10⁹/L", type: "number" },
    ],
    compute(v) {
      const age = num(v.age), ast = num(v.ast), alt = num(v.alt), plt = num(v.plt);
      if (age == null || ast == null || alt == null || plt == null || plt <= 0 || alt <= 0) return null;
      const fib = round((age * ast) / (plt * Math.sqrt(alt)), 2);
      const tone = fib > 2.67 ? "warn" : "good";
      return { value: fib, unit: "", tone, note: "< 1.3 low, > 2.67 advanced fibrosis likely." };
    },
  },
  {
    id: "osmgap", name: "Osmolar gap", blurb: "Measured − calculated osmolality.",
    inputs: [
      { key: "na", label: "Sodium", unit: "mmol/L", type: "number" },
      { key: "glu", label: "Glucose", unit: "mmol/L", type: "number" },
      { key: "urea", label: "Urea", unit: "mmol/L", type: "number" },
      { key: "meas", label: "Measured osmolality", unit: "mOsm/kg", type: "number" },
    ],
    compute(v) {
      const na = num(v.na), glu = num(v.glu), urea = num(v.urea), meas = num(v.meas);
      if (na == null || glu == null || urea == null || meas == null) return null;
      const calc = 2 * na + glu + urea;
      const gap = round(meas - calc, 1);
      const tone = gap > 10 ? "warn" : "good";
      return { value: gap, unit: "mOsm/kg", tone, note: "> 10 → unmeasured osmoles (e.g. toxic alcohols, mannitol)." };
    },
  },
  {
    id: "winters", name: "Winters' formula", blurb: "Expected pCO₂ in metabolic acidosis.",
    inputs: [{ key: "hco3", label: "Bicarbonate", unit: "mmol/L", type: "number" }],
    compute(v) {
      const hco3 = num(v.hco3);
      if (hco3 == null) return null;
      const mmHg = 1.5 * hco3 + 8;
      const kpa = round(mmHg / 7.5, 1);
      return { value: kpa, unit: "kPa", tone: "info",
        note: `Expected pCO₂ ≈ ${kpa} kPa (${round(mmHg, 0)} mmHg) ± 0.3. Lower measured → added respiratory alkalosis; higher → added respiratory acidosis.` };
    },
  },
  {
    id: "nadeficit", name: "Sodium deficit", blurb: "For hyponatraemia correction planning.",
    inputs: [
      { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
      { key: "cur", label: "Current sodium", unit: "mmol/L", type: "number" },
      { key: "tgt", label: "Target sodium", unit: "mmol/L", type: "number" },
    ],
    compute(v) {
      const wt = num(v.wt), cur = num(v.cur), tgt = num(v.tgt);
      if (wt == null || cur == null || tgt == null || !v.sex) return null;
      const tbw = wt * (v.sex === "female" ? 0.5 : 0.6);
      const deficit = round(tbw * (tgt - cur), 0);
      return { value: deficit, unit: "mmol", tone: "warn",
        note: "Correct SLOWLY (≤ 8–10 mmol/L per 24h) to avoid osmotic demyelination." };
    },
  },
  {
    id: "parkland", name: "Parkland formula (burns)", blurb: "First-24h crystalloid.",
    inputs: [
      { key: "wt", label: "Weight", unit: "kg", type: "number" },
      { key: "tbsa", label: "% TBSA burned", unit: "%", type: "number" },
    ],
    compute(v) {
      const wt = num(v.wt), tbsa = num(v.tbsa);
      if (wt == null || tbsa == null) return null;
      const total = 4 * wt * tbsa;
      const first8 = round(total / 2, 0);
      return { value: round(total, 0), unit: "mL / 24h", tone: "info",
        note: `≈ ${first8} mL in the first 8h (from time of burn), remainder over 16h. Titrate to urine output.` };
    },
  },
  {
    id: "fena", name: "Fractional excretion of Na (FENa)", blurb: "Differentiates prerenal AKI vs ATN.",
    inputs: [
      { key: "una", label: "Urine sodium", unit: "mmol/L", type: "number" },
      { key: "pna", label: "Plasma sodium", unit: "mmol/L", type: "number" },
      { key: "ucr", label: "Urine creatinine", unit: "µmol/L", type: "number" },
      { key: "pcr", label: "Plasma creatinine", unit: "µmol/L", type: "number" },
    ],
    compute(v) {
      const una = num(v.una), pna = num(v.pna), ucr = num(v.ucr), pcr = num(v.pcr);
      if (una == null || pna == null || ucr == null || pcr == null || pna === 0 || ucr === 0) return null;
      const fena = round(((una * pcr) / (pna * ucr)) * 100, 2);
      const tone = fena < 1 ? "warn" : "info";
      return { value: fena, unit: "%", tone,
        note: "< 1% prerenal; > 2% ATN. Unreliable on diuretics — consider FEUrea." };
    },
  },
  {
    id: "aagrad", name: "A–a gradient", blurb: "Alveolar–arterial O₂ gradient (kPa, sea level).",
    inputs: [
      { key: "fio2", label: "FiO₂ (fraction, e.g. 0.21)", type: "number" },
      { key: "paco2", label: "PaCO₂", unit: "kPa", type: "number" },
      { key: "pao2", label: "PaO₂", unit: "kPa", type: "number" },
      { key: "age", label: "Age (optional)", unit: "years", type: "number" },
    ],
    compute(v) {
      const fio2 = num(v.fio2), paco2 = num(v.paco2), pao2 = num(v.pao2);
      if (fio2 == null || paco2 == null || pao2 == null) return null;
      const pAO2 = fio2 * 95 - paco2 / 0.8;   // Patm−PH2O ≈ 95 kPa at sea level
      const aa = round(pAO2 - pao2, 1);
      const age = num(v.age);
      const expected = age != null ? round((age / 4 + 4) / 7.5, 1) : null;
      const tone = expected != null ? (aa > expected ? "warn" : "good") : "info";
      return { value: aa, unit: "kPa", tone,
        note: expected != null ? `Expected ≈ ${expected} kPa for age. Raised → V/Q mismatch, shunt or diffusion defect.`
          : "Compare to age-expected (≈ (age/4 + 4)/7.5 kPa)." };
    },
  },
  {
    id: "gad7", name: "GAD-7 (anxiety)", blurb: "Over the last 2 weeks (0–3 each).",
    inputs: ["Feeling nervous/anxious/on edge", "Not being able to stop/control worrying",
      "Worrying too much about different things", "Trouble relaxing",
      "Being so restless it's hard to sit still", "Becoming easily annoyed/irritable",
      "Feeling afraid as if something awful might happen"].map((label, i) => ({
      key: `g${i}`, label, type: "select", options: [
        { value: 0, label: "0 — not at all" }, { value: 1, label: "1 — several days" },
        { value: 2, label: "2 — more than half the days" }, { value: 3, label: "3 — nearly every day" }] })),
    compute(v) {
      const parts = [0, 1, 2, 3, 4, 5, 6].map((i) => num(v[`g${i}`]));
      if (parts.some((p) => p == null)) return null;
      const s = parts.reduce((a, b) => a + b, 0);
      const sev = s >= 15 ? "severe" : s >= 10 ? "moderate" : s >= 5 ? "mild" : "minimal";
      const tone = s >= 10 ? "warn" : "good";
      return { value: s, unit: "/21", tone, note: `${sev} anxiety symptoms (≥10 clinically significant).` };
    },
  },
  {
    id: "phq9", name: "PHQ-9 (depression)", blurb: "Over the last 2 weeks (0–3 each).",
    inputs: ["Little interest or pleasure", "Feeling down/depressed/hopeless",
      "Sleep problems", "Tired / little energy", "Poor appetite or overeating",
      "Feeling bad about yourself", "Trouble concentrating",
      "Moving/speaking slowly or being restless", "Thoughts of self-harm"].map((label, i) => ({
      key: `p${i}`, label, type: "select", options: [
        { value: 0, label: "0 — not at all" }, { value: 1, label: "1 — several days" },
        { value: 2, label: "2 — more than half the days" }, { value: 3, label: "3 — nearly every day" }] })),
    compute(v) {
      const parts = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => num(v[`p${i}`]));
      if (parts.some((p) => p == null)) return null;
      const s = parts.reduce((a, b) => a + b, 0);
      const sev = s >= 20 ? "severe" : s >= 15 ? "moderately severe" : s >= 10 ? "moderate" : s >= 5 ? "mild" : "minimal";
      const tone = s >= 10 ? "warn" : "good";
      const q9 = num(v.p8);
      return { value: s, unit: "/27", tone,
        note: `${sev} (≥10 warrants action).${q9 ? " Item 9 positive — assess suicide risk." : ""}` };
    },
  },
  {
    id: "naegele", name: "EDD (Naegele's rule)", blurb: "Estimated due date + gestational age from LMP.",
    inputs: [{ key: "lmp", label: "First day of last menstrual period", type: "date" }],
    compute(v) {
      if (!v.lmp) return null;
      const lmp = new Date(v.lmp + "T00:00:00");
      if (Number.isNaN(lmp.getTime())) return null;
      const edd = new Date(lmp.getTime() + 280 * 86400000);
      const days = Math.floor((Date.now() - lmp.getTime()) / 86400000);
      const wk = Math.floor(days / 7), d = ((days % 7) + 7) % 7;
      const ga = days >= 0 && days <= 300 ? ` · GA today ≈ ${wk}⁺${d} weeks` : "";
      const iso = edd.toISOString().slice(0, 10);
      return { value: iso, unit: "EDD", tone: "info",
        note: `LMP + 280 days (40 weeks)${ga}. Confirm by early ultrasound where dating is uncertain.` };
    },
  },
  {
    id: "alvarado", name: "Alvarado score (appendicitis)", blurb: "MANTRELS — likelihood of acute appendicitis.",
    inputs: [
      { key: "mig", label: "Migratory RIF pain", type: "check" },
      { key: "anor", label: "Anorexia", type: "check" },
      { key: "naus", label: "Nausea / vomiting", type: "check" },
      { key: "tend", label: "RIF tenderness (2)", type: "check" },
      { key: "reb", label: "Rebound tenderness", type: "check" },
      { key: "temp", label: "Fever ≥37.3 °C", type: "check" },
      { key: "leuk", label: "Leucocytosis >10 (2)", type: "check" },
      { key: "shift", label: "Neutrophil left shift", type: "check" },
    ],
    compute(v) {
      const s = (v.mig ? 1 : 0) + (v.anor ? 1 : 0) + (v.naus ? 1 : 0) + (v.tend ? 2 : 0)
        + (v.reb ? 1 : 0) + (v.temp ? 1 : 0) + (v.leuk ? 2 : 0) + (v.shift ? 1 : 0);
      const band = s >= 7 ? "appendicitis likely — surgical review" : s >= 5 ? "compatible — observe / image"
        : "unlikely — consider discharge with safety-net";
      const tone = s >= 7 ? "bad" : s >= 5 ? "warn" : "good";
      return { value: s, unit: "/10", tone, note: band + "." };
    },
  },
  {
    id: "gbs", name: "Glasgow-Blatchford (upper GI bleed)", blurb: "Pre-endoscopy risk; 0 may allow outpatient care.",
    inputs: [
      { key: "urea", label: "Urea", unit: "mmol/L", type: "number" },
      { key: "hb", label: "Haemoglobin", unit: "g/dL", type: "number" },
      { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
      { key: "sbp", label: "Systolic BP", unit: "mmHg", type: "number" },
      { key: "hr", label: "Pulse ≥100", type: "check" },
      { key: "melaena", label: "Melaena", type: "check" },
      { key: "syncope", label: "Syncope", type: "check" },
      { key: "liver", label: "Hepatic disease", type: "check" },
      { key: "cardiac", label: "Cardiac failure", type: "check" },
    ],
    compute(v) {
      const urea = num(v.urea), hb = num(v.hb), sbp = num(v.sbp);
      if (urea == null || hb == null || sbp == null || !v.sex) return null;
      let s = 0;
      s += urea >= 25 ? 6 : urea >= 10 ? 4 : urea >= 8 ? 3 : urea >= 6.5 ? 2 : 0;
      const female = v.sex === "female";
      if (female) s += hb < 10 ? 6 : hb < 12 ? 1 : 0;
      else s += hb < 10 ? 6 : hb < 12 ? 3 : hb < 13 ? 1 : 0;
      s += sbp < 90 ? 3 : sbp < 100 ? 2 : sbp < 110 ? 1 : 0;
      s += (v.hr ? 1 : 0) + (v.melaena ? 1 : 0) + (v.syncope ? 2 : 0)
        + (v.liver ? 2 : 0) + (v.cardiac ? 2 : 0);
      const tone = s === 0 ? "good" : s <= 5 ? "warn" : "bad";
      return { value: s, unit: "pts", tone,
        note: s === 0 ? "Very low risk — consider outpatient management." : "Score >0 — admit / endoscopy per pathway." };
    },
  },
  {
    id: "ciwa", name: "CIWA-Ar (alcohol withdrawal)", blurb: "10 items · guides symptom-triggered benzodiazepines.",
    inputs: [
      ...["Nausea/vomiting", "Tremor", "Paroxysmal sweats", "Anxiety",
        "Agitation", "Tactile disturbances", "Auditory disturbances",
        "Visual disturbances", "Headache"].map((label, i) => ({
        key: `c${i}`, label, type: "select",
        options: [0, 1, 2, 3, 4, 5, 6, 7].map((n) => ({ value: n, label: String(n) })) })),
      { key: "orient", label: "Orientation / clouding of sensorium", type: "select",
        options: [0, 1, 2, 3, 4].map((n) => ({ value: n, label: String(n) })) },
    ],
    compute(v) {
      const parts = [...Array(9).keys()].map((i) => num(v[`c${i}`]));
      const ori = num(v.orient);
      if (parts.some((p) => p == null) || ori == null) return null;
      const s = parts.reduce((a, b) => a + b, 0) + ori;
      const band = s >= 20 ? "severe — high seizure/DT risk" : s >= 8 ? "moderate — medicate"
        : "minimal — monitor";
      const tone = s >= 20 ? "bad" : s >= 8 ? "warn" : "good";
      return { value: s, unit: "/67", tone, note: `${band}. ≥8–10 usually triggers benzodiazepine dosing.` };
    },
  },
  {
    id: "ome", name: "Opioid → oral morphine equivalent", blurb: "Approximate daily oral morphine milligram equivalent (OME).",
    inputs: [
      { key: "drug", label: "Opioid", type: "select", options: [
        { value: "morphine", label: "Morphine (oral)" },
        { value: "morphine_iv", label: "Morphine (IV/SC)" },
        { value: "codeine", label: "Codeine (oral)" },
        { value: "tramadol", label: "Tramadol (oral)" },
        { value: "oxycodone", label: "Oxycodone (oral)" },
        { value: "hydromorphone", label: "Hydromorphone (oral)" },
        { value: "fentanyl_patch", label: "Fentanyl patch (µg/hr)" }] },
      { key: "dose", label: "Total daily dose", unit: "mg/day (or µg/hr patch)", type: "number" },
    ],
    compute(v) {
      const dose = num(v.dose);
      if (dose == null || !v.drug) return null;
      const F = { morphine: 1, morphine_iv: 3, codeine: 0.15, tramadol: 0.1,
        oxycodone: 1.5, hydromorphone: 5, fentanyl_patch: 2.4 };
      const ome = round(dose * F[v.drug], 0);
      const tone = ome >= 90 ? "bad" : ome >= 50 ? "warn" : "good";
      return { value: ome, unit: "mg OME/day", tone,
        note: "Approximate — reduce 25–50% for incomplete cross-tolerance when rotating. ≥90 mg/day: heightened risk." };
    },
  },
  {
    id: "wells_dvt", name: "Wells score (DVT)", blurb: "Pre-test probability of deep vein thrombosis.",
    inputs: [
      { key: "cancer", label: "Active cancer", type: "check" },
      { key: "paresis", label: "Paralysis/paresis or recent immobilisation of leg", type: "check" },
      { key: "bedrid", label: "Bedridden ≥3d or major surgery <12wk", type: "check" },
      { key: "tender", label: "Localised tenderness along deep veins", type: "check" },
      { key: "swollen", label: "Entire leg swollen", type: "check" },
      { key: "calf", label: "Calf swelling >3 cm vs other leg", type: "check" },
      { key: "pitting", label: "Pitting oedema (symptomatic leg)", type: "check" },
      { key: "collat", label: "Collateral superficial veins (non-varicose)", type: "check" },
      { key: "prior", label: "Previously documented DVT", type: "check" },
      { key: "alt", label: "Alternative diagnosis as likely (−2)", type: "check" },
    ],
    compute(v) {
      const keys = ["cancer", "paresis", "bedrid", "tender", "swollen", "calf",
        "pitting", "collat", "prior"];
      const s = keys.reduce((a, k) => a + (v[k] ? 1 : 0), 0) - (v.alt ? 2 : 0);
      const band = s >= 2 ? "DVT likely — proceed to ultrasound" : "DVT unlikely — D-dimer to exclude";
      const tone = s >= 2 ? "bad" : "good";
      return { value: s, unit: "pts", tone, note: band + "." };
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
  ["INR (on warfarin, target)", "2.0–3.0", ""], ["TSH", "0.27–4.2", "mIU/L"],
  ["HbA1c (target)", "< 7 (individualise)", "%"],
  ["Arterial pH", "7.35–7.45", ""], ["pCO₂", "4.7–6.0", "kPa"],
  ["pO₂", "11–13", "kPa"], ["Bicarbonate (ABG)", "22–26", "mmol/L"],
  ["Lactate", "0.5–2.2", "mmol/L"],
  ["LDL (high-risk target)", "< 1.8", "mmol/L"], ["HDL (M / F)", "> 1.0 / > 1.2", "mmol/L"],
  ["Triglycerides", "< 1.7", "mmol/L"], ["Ferritin (M / F)", "30–400 / 15–150", "µg/L"],
  ["Vitamin D (25-OH)", "> 50", "nmol/L"],
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
            }, [h("option", { value: "" }, "—"), ...inp.options.map((o) => {
              const val = typeof o === "object" ? String(o.value) : o;
              const label = typeof o === "object" ? o.label : o;
              return h("option", { value: val, selected: String(values[inp.key]) === val }, label);
            })]));
        }
        if (inp.type === "date") {
          return h("label.calc-field", {},
            h("span.calc-label", {}, inp.label),
            h("input.input", {
              type: "date", value: values[inp.key] ?? "",
              oninput: (ev) => { values[inp.key] = ev.target.value; persist(); paint(); },
            }));
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
