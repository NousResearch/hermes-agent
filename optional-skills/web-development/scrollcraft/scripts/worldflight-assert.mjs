#!/usr/bin/env node
/**
 * Worldflight rig assertions. Not a contact sheet: these are the mechanical
 * claims the mode makes, checked one at a time.
 *
 *   node lab/worldflight-assert.mjs --url http://localhost:4520
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const { chromium } = createRequire(path.join(process.cwd(), "package.json"))("playwright-core");

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i > -1 && argv[i + 1] ? argv[i + 1] : d; };
const URL = arg("--url", "http://localhost:4520");

const CHROME = [
  process.env.SCROLLCRAFT_CHROME,
  // Windows
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  // macOS
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  // Linux
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
].find((p) => p && fs.existsSync(p));

let pass = 0, fail = 0;
const ok = (name, cond, note = "") => {
  if (cond) { pass++; console.log(`  PASS  ${name}${note ? "  " + note : ""}`); }
  else { fail++; console.log(`  FAIL  ${name}${note ? "  " + note : ""}`); }
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });

// ============================================================ desktop ======
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
  page.on("pageerror", (e) => errs.push(String(e)));
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("html.sc-ready", { timeout: 15000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(900);

  const geom = await page.evaluate(() => {
    const segs = [...document.querySelectorAll("[data-sc-segment]")];
    const total = segs.reduce((s, el) => s + (parseFloat(el.dataset.scW) || 1.3), 0);
    const spacer = document.querySelector("[data-sc-spacer]");
    const stage = document.querySelector("[data-sc-world]");
    return {
      vh: innerHeight, total,
      weights: segs.map((el) => parseFloat(el.dataset.scW) || 1.3),
      spacerH: spacer.getBoundingClientRect().height,
      wantH: Math.round((total + 1) * innerHeight),
      docH: document.body.scrollHeight,
      stagePos: getComputedStyle(stage).position,
      stageRect: (({ top, left, width, height }) => ({ top, left, width, height }))(stage.getBoundingClientRect()),
      copyPos: getComputedStyle(document.querySelector("[data-sc-world-copy]")).position,
      seam: parseFloat(document.querySelector('[data-sc-mode="worldflight"]').dataset.scSeam),
    };
  });
  console.log(`\nDESKTOP  vh=${geom.vh}  weights=[${geom.weights}]  total=${geom.total}vh`);

  ok("spacer height = (sum weights + 1) x vh",
    Math.abs(geom.spacerH - geom.wantH) <= 1, `got ${Math.round(geom.spacerH)}px want ${geom.wantH}px`);
  ok("document scroll height is the spacer",
    Math.abs(geom.docH - geom.wantH) <= 2, `doc ${geom.docH}px`);
  ok("stage is position:fixed and fills the viewport",
    geom.stagePos === "fixed" && geom.stageRect.height === geom.vh && geom.stageRect.top === 0);
  ok("copy layer is position:fixed", geom.copyPos === "fixed");

  // Nothing in flow but the spacer: at any scroll position, the only static or
  // relative element that extends past the fold is the spacer and its ancestors.
  const flow = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll("body *").forEach((el) => {
      const cs = getComputedStyle(el);
      if (cs.position === "fixed" || cs.position === "absolute") return;
      if (el.hasAttribute("data-sc-spacer") || el.hasAttribute("data-sc-mode")) return;
      const r = el.getBoundingClientRect();
      if (r.height > 4 && r.bottom > innerHeight + 4) {
        bad.push(el.tagName + "." + (el.className || "").toString().slice(0, 30));
      }
    });
    return bad;
  });
  ok("nothing in document flow but the spacer", flow.length === 0, flow.join(", "));

  // ---- lerp convergence, inside one leg ----------------------------------
  const seg0 = geom.weights[0];
  const A = Math.round(0.10 * seg0 * geom.vh);
  const B = Math.round(0.85 * seg0 * geom.vh);
  await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), A);
  await page.waitForTimeout(1400);

  const trace = await page.evaluate(({ y }) => new Promise((res) => {
    const v = document.querySelectorAll("[data-sc-segment] video")[0];
    const clip = window.__sc.clips.find((c) => c.el === v);
    const s = [];
    scrollTo({ top: y, behavior: "instant" });
    let n = 0;
    (function f() {
      s.push(+v.currentTime.toFixed(4));
      if (++n < 70) requestAnimationFrame(f);
      else res({ s, target: clip.target * (v.duration || 1), dur: v.duration, lerp: clip.lerp });
    })();
  }), { y: B });

  const seq = trace.s;
  const start = seq[0], end = seq[seq.length - 1];
  const distinct = [...new Set(seq)];
  const monotone = seq.every((v, i) => i === 0 || v >= seq[i - 1] - 1e-6);
  const overshoot = Math.max(...seq) - trace.target;
  console.log(`  playhead ${start.toFixed(3)}s -> ${end.toFixed(3)}s  target ${trace.target.toFixed(3)}s  lerp ${trace.lerp}  ${distinct.length} distinct values`);
  ok("playhead is lerped, not written 1:1", distinct.length >= 4, `${distinct.length} distinct steps across 70 frames`);
  ok("playhead is strictly monotone toward its target", monotone);
  ok("playhead does not overshoot", overshoot <= 0.02, `max excess ${overshoot.toFixed(4)}s`);
  ok("playhead converges on the target", Math.abs(end - trace.target) < 0.05,
    `residual ${Math.abs(end - trace.target).toFixed(4)}s`);
  ok("lerp rate is the 0.18 default", Math.abs(trace.lerp - 0.18) < 1e-9);

  // ---- deadband: no seek thrash while the page is still -------------------
  await page.waitForTimeout(900);
  const seeks = await page.evaluate(() => new Promise((res) => {
    let n = 0;
    const vs = [...document.querySelectorAll("video")];
    const h = () => n++;
    vs.forEach((v) => v.addEventListener("seeked", h));
    setTimeout(() => { vs.forEach((v) => v.removeEventListener("seeked", h)); res(n); }, 1000);
  }));
  ok("deadband holds: no seeks over 1s of stillness", seeks === 0, `${seeks} seeked events`);

  // ---- crossfade band ----------------------------------------------------
  const boundary = seg0 * geom.vh;
  const half = (geom.seam / 2) * geom.vh;
  const band = [];
  for (let i = 0; i < 5; i++) {
    const y = Math.round(boundary - half + (2 * half) * (i / 4));
    await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), y);
    await page.waitForTimeout(120);
    band.push(await page.evaluate(() => {
      const segs = [...document.querySelectorAll("[data-sc-segment]")];
      return segs.map((s) => +(parseFloat(getComputedStyle(s).opacity) || 0).toFixed(4));
    }));
  }
  const incoming = band.map((b) => b[1]);
  const outgoing = band.map((b) => b[0]);
  console.log(`  seam band  incoming=[${incoming}]  outgoing=[${outgoing}]`);
  ok("incoming leg opacity is strictly monotone across the seam",
    incoming.every((v, i) => i === 0 || v > incoming[i - 1]));
  ok("incoming leg starts at 0 and ends at 1", incoming[0] <= 0.02 && incoming[4] >= 0.98);
  // The outgoing leg must hold at full strength for the whole dissolve and only
  // release once the incoming one covers it completely. A crossfade where BOTH
  // sides are partly transparent shows the page ground through the middle of
  // the seam, which is the flash this mode exists to remove.
  ok("outgoing leg holds full opacity while the incoming one is still arriving",
    incoming.every((v, i) => v >= 0.999 || outgoing[i] >= 0.999), `[${outgoing}]`);
  ok("outgoing leg only releases once it is fully covered",
    outgoing[4] === 0 && incoming[4] >= 0.999);

  // ---- copy transform cap ------------------------------------------------
  const maxT = [];
  for (let i = 0; i <= 12; i++) {
    const y = Math.round((geom.docH - geom.vh) * (i / 12));
    await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), y);
    await page.waitForTimeout(60);
    maxT.push(await page.evaluate(() => {
      let m = 0;
      document.querySelectorAll("[data-sc-copy]").forEach((el) => {
        const t = new DOMMatrixReadOnly(getComputedStyle(el).transform);
        m = Math.max(m, Math.abs(t.m42), Math.abs(t.m41));
      });
      return +m.toFixed(2);
    }));
  }
  const worstT = Math.max(...maxT);
  ok("copy never translates past 4vh", worstT <= geom.vh * 0.04 + 1,
    `worst ${worstT}px, cap ${(geom.vh * 0.04).toFixed(0)}px`);

  // ---- every leg reaches full opacity, every clip leaves its poster -------
  const reach = {};
  for (let i = 0; i <= 24; i++) {
    const y = Math.round((geom.docH - geom.vh) * (i / 24));
    await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), y);
    await page.waitForTimeout(260);
    const s = await page.evaluate(() => [...document.querySelectorAll("[data-sc-segment]")].map((el, i) => ({
      i, op: parseFloat(getComputedStyle(el).opacity) || 0,
      painted: el.classList.contains("sc-has-clip"),
    })));
    s.forEach((x) => {
      reach[x.i] = reach[x.i] || { op: 0, painted: false };
      reach[x.i].op = Math.max(reach[x.i].op, x.op);
      reach[x.i].painted = reach[x.i].painted || x.painted;
    });
  }
  ok("every leg reaches full opacity",
    Object.values(reach).every((r) => r.op >= 0.99),
    Object.entries(reach).map(([i, r]) => `${i}:${r.op.toFixed(2)}`).join(" "));
  ok("every leg paints a real frame (no clip stuck on poster)",
    Object.values(reach).every((r) => r.painted),
    Object.entries(reach).map(([i, r]) => `${i}:${r.painted ? "clip" : "POSTER"}`).join(" "));

  ok("no console errors", errs.length === 0, errs.slice(0, 3).join(" | "));
  await page.close();
}

// ==================================================== reduced motion =======
{
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, reducedMotion: "reduce",
  });
  const media = [];
  page.on("request", (r) => { if (/\.(mp4|webm)(\?|$)/.test(r.url())) media.push(r.url()); });
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("html.sc-ready", { timeout: 15000 });
  await page.waitForTimeout(800);
  console.log("\nREDUCED MOTION");

  const rmSeen = { legs: {}, copy: {} };
  for (let i = 0; i <= 20; i++) {
    const h = await page.evaluate(() => document.body.scrollHeight - innerHeight);
    await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), Math.round(h * (i / 20)));
    await page.waitForTimeout(90);
    const s = await page.evaluate(() => ({
      legs: [...document.querySelectorAll("[data-sc-segment]")].map((el) => ({
        op: parseFloat(getComputedStyle(el).opacity) || 0,
        poster: (() => { const p = el.querySelector(".sc-world__poster"); return p ? getComputedStyle(p).transform : "none"; })(),
      })),
      copy: [...document.querySelectorAll("[data-sc-copy]")].map((el) => ({
        t: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 24),
        op: parseFloat(getComputedStyle(el).opacity) || 0,
        tf: getComputedStyle(el).transform,
      })),
    }));
    s.legs.forEach((l, k) => {
      rmSeen.legs[k] = rmSeen.legs[k] || { op: 0, tf: new Set() };
      rmSeen.legs[k].op = Math.max(rmSeen.legs[k].op, l.op);
      rmSeen.legs[k].tf.add(l.poster);
    });
    s.copy.forEach((c) => {
      rmSeen.copy[c.t] = rmSeen.copy[c.t] || { op: 0, tf: new Set() };
      rmSeen.copy[c.t].op = Math.max(rmSeen.copy[c.t].op, c.op);
      rmSeen.copy[c.t].tf.add(c.tf);
    });
  }
  await page.waitForTimeout(500);

  ok("no clip is ever fetched", media.length === 0, media.join(", "));
  ok("every leg's poster still reaches full opacity",
    Object.values(rmSeen.legs).every((l) => l.op >= 0.99),
    Object.entries(rmSeen.legs).map(([i, l]) => `${i}:${l.op.toFixed(2)}`).join(" "));
  ok("no poster transform",
    Object.values(rmSeen.legs).every((l) => [...l.tf].every((t) => t === "none")));
  ok("every copy block still reaches full opacity",
    Object.values(rmSeen.copy).every((c) => c.op >= 0.99),
    Object.entries(rmSeen.copy).map(([t, c]) => `${c.op.toFixed(2)} "${t}"`).join(" | "));
  ok("no copy transform",
    Object.values(rmSeen.copy).every((c) => [...c.tf].every((t) => t === "none")));
  await page.close();
}

await browser.close();
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
