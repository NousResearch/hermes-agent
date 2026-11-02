#!/usr/bin/env node
/**
 * scrollcraft verification harness: shoot the page's own scroll.
 *
 * Walks the page in N evenly spaced scroll positions, waits for the scrub video
 * to actually settle at each one, screenshots it, and reports what the engine
 * thinks is on screen. Then tiles the frames into one contact sheet, because a
 * dead middle only shows up in contiguous frames: any single screenshot is a
 * frame the transition may not survive.
 *
 *   node shoot.mjs --url http://localhost:4500 --out lab/shots --steps 12
 *   node shoot.mjs --url ... --width 375 --height 812 --out lab/mobile
 *   node shoot.mjs --url ... --reduced-motion --out lab/reduced
 *
 * Uses the INSTALLED Chrome, not bundled Chromium: Chromium ships without the
 * h264 decoder, so every scrub clip would silently fail to paint and the run
 * would "pass" against posters.
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// The skill lives outside the project it is building, so resolve playwright
// from the BUILD project's node_modules (cwd), not from next to this file.
// Run `npm i playwright-core` in the build project once.
let chromium;
try {
  ({ chromium } = createRequire(path.join(process.cwd(), "package.json"))("playwright-core"));
} catch {
  console.error("playwright-core not found. Run this in the build project after:\n  npm i playwright-core");
  process.exit(1);
}

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i > -1 && argv[i + 1] ? argv[i + 1] : d; };
const has = (n) => argv.includes(n);

const URL = arg("--url", "http://localhost:4500");
const OUT = path.resolve(arg("--out", "lab/shots"));
const STEPS = parseInt(arg("--per-act", arg("--steps", "6")), 10);  // samples PER ACT
const W = parseInt(arg("--width", "1440"), 10);
const H = parseInt(arg("--height", "900"), 10);
const REDUCED = has("--reduced-motion");

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

if (!CHROME) {
  console.error("No installed Chrome found. Set SCROLLCRAFT_CHROME to its path.");
  process.exit(1);
}

fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
  reducedMotion: REDUCED ? "reduce" : "no-preference",
});

const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));
const failed = [];
page.on("requestfailed", (r) => failed.push(`${r.failure()?.errorText} ${r.url()}`));

// Not networkidle: the engine keeps clips in flight as you scroll, and a
// webfont connection can stay open, so idle may never arrive. Wait for the
// engine's own ready signal and for the faces to land, since line splitting
// measures real line boxes and is wrong before the real face is applied.
await page.goto(URL, { waitUntil: "domcontentloaded" });
await page.waitForSelector("html.sc-ready", { timeout: 15000 });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(700);

const doc = await page.evaluate(() => {
  const world = document.querySelector('[data-sc-mode="worldflight"]');
  return {
    height: document.body.scrollHeight,
    vh: innerHeight,
    acts: [...document.querySelectorAll("[data-sc-act]")].map((a) => a.dataset.scAct),
    world: world
      ? {
          seam: parseFloat(world.dataset.scSeam) || 0.12,
          segs: [...world.querySelectorAll("[data-sc-segment]")].map((s) => ({
            w: parseFloat(s.dataset.scW) || 1.3,
            linger: parseFloat(s.dataset.scLinger) || 0,
            label: s.dataset.scWaypoint || "",
          })),
        }
      : null,
  };
});
const WORLD = doc.world;
const maxScroll = doc.height - doc.vh;

if (WORLD) {
  const total = WORLD.segs.reduce((s, g) => s + g.w, 0);
  console.log(`page: worldflight, ${WORLD.segs.length} legs over ${total.toFixed(2)}vh ` +
    `(track ${(doc.height / doc.vh).toFixed(1)} viewport-heights), seam ${WORLD.seam}vh`);
  console.log(`  legs: ${WORLD.segs.map((g, i) => `${i}:${g.label || "-"}@${g.w}vh`).join("  ")}`);
} else {
  console.log(`page: ${(doc.height / doc.vh).toFixed(1)} viewport-heights, acts: ${doc.acts.join(" > ")}`);
}

// Wait for the playhead to ARRIVE, not merely to stop seeking. The engine lerps
// currentTime toward a target on its own rAF loop, so after any scroll jump
// there is a stretch of ~15 frames during which every clip on the page is
// somewhere it will never be again. Screenshot in that window and the sheet is a
// set of frames the reader is never shown, the dead-scroll comparison runs on
// mid-lerp noise, and the whole run is unrepeatable.
async function settle(timeout = 4000) {
  const t0 = Date.now();
  let last = null;
  for (;;) {
    const now = await page.evaluate(() => {
      const insts = (window.ScrollCraft && window.ScrollCraft.instances) || [];
      const clips = [].concat(...insts.map((i) => i.clips || [])).filter((c) => c.ready);
      // Pages on an older engine expose no instances; fall back to watching
      // currentTime go quiet, which reaches the same state more slowly.
      if (clips.length) {
        const arrived = clips.every((c) => Math.abs(c.cur - c.target) < 0.002 && !c.el.seeking);
        return arrived ? "arrived" : "moving";
      }
      return [...document.querySelectorAll("video[data-sc-scrub]")]
        .map((v) => (v.seeking ? "seeking" : v.currentTime.toFixed(3))).join("|");
    });
    if (now === "arrived") return true;
    if (now !== "moving" && now === last && !now.includes("seeking")) return true;
    if (Date.now() - t0 > timeout) return false;
    last = now;
    await page.waitForTimeout(60);
  }
}

// Sample WITHIN each act, not uniformly down the document. Uniform sampling
// distributes positions by page length, so a short act gets one sample that
// lands wherever it lands, and adding a section elsewhere silently moves every
// sample. That produces "this cue never reaches full opacity" reports that come
// and go with unrelated edits. Per-act sampling hits the same fractions of
// every act every run, so the findings mean something.
//
// A worldflight has no acts to sample within; its unit is the leg, and its
// geometry lives entirely in the weights, so positions are computed from the
// track rather than measured off the DOM. Both sides of every seam are added on
// top: the crossfade is the frame this mode is judged on, and it occupies about
// a tenth of a viewport, so uniform sampling steps straight over it.
const positions = WORLD ? await page.evaluate((perSeg) => {
  const root = document.querySelector('[data-sc-mode="worldflight"]');
  const segs = [...root.querySelectorAll("[data-sc-segment]")];
  const top = root.getBoundingClientRect().top + scrollY;
  const seam = parseFloat(root.dataset.scSeam) || 0.12;
  const fracs = Array.from({ length: perSeg }, (_, i) => (perSeg === 1 ? 0.5 : i / (perSeg - 1)));
  const out = [];
  let c = 0;
  segs.forEach((s, i) => {
    const w = parseFloat(s.dataset.scW) || 1.3;
    fracs.forEach((f) => {
      const p = 0.02 + f * 0.96;
      out.push(Math.round(top + (c + w * p) * innerHeight));
    });
    c += w;
    if (i < segs.length - 1) {
      [-0.5, -0.2, 0.2, 0.5].forEach((k) => out.push(Math.round(top + (c + k * seam) * innerHeight)));
    }
  });
  const max = document.body.scrollHeight - innerHeight;
  out.push(0, max);
  return [...new Set(out.map((y) => Math.max(0, Math.min(max, y))))].sort((a, b) => a - b);
}, Math.max(2, Math.round(STEPS))) : await page.evaluate((perAct) => {
  const out = [];
  const fracs = Array.from({ length: perAct }, (_, i) => (perAct === 1 ? 0.5 : i / (perAct - 1)));
  document.querySelectorAll("[data-sc-act]").forEach((el) => {
    const top = el.getBoundingClientRect().top + scrollY;
    const h = el.offsetHeight;
    const pinned = ["scrub", "pin", "pan"].includes(el.dataset.scAct);
    fracs.forEach((f) => {
      // Nudge off the exact endpoints: p=0 and p=1 sit on the seam between two
      // acts, where which one you are "in" is ambiguous.
      const p = 0.02 + f * 0.96;
      out.push(Math.round(pinned ? top + (h - innerHeight) * p : top - innerHeight + (h + innerHeight) * p));
    });
    // A pinned stage is on screen for a viewport BEFORE its pinned travel begins
    // and a viewport AFTER it ends, and the loop above samples only inside the
    // travel. Those two slides are exactly where a clip mapped to pinned
    // progress sits frozen on its first or last frame, so not sampling them is
    // why a frozen clip could pass this harness. Sample them.
    if (el.dataset.scAct === "scrub") {
      // `v` is the fraction of the viewport the stage covers at that position.
      // Sample the part of each slide where the stage is still MOSTLY on screen,
      // because that is where a frozen frame is conspicuous, and because the
      // frozen-clip check needs consecutive samples that are both well past its
      // visibility gate before it will call anything.
      [0.6, 0.75, 0.9].forEach((v) => {
        out.push(Math.round(top - innerHeight * (1 - v)));  // sliding in
        out.push(Math.round(top + h - innerHeight * v));    // sliding out
      });
    }
  });
  const max = document.body.scrollHeight - innerHeight;
  out.push(max);
  return [...new Set(out.map((y) => Math.max(0, Math.min(max, y))))].sort((a, b) => a - b);
}, Math.max(2, Math.round(STEPS)));

const report = [];
for (let i = 0; i < positions.length; i++) {
  const y = positions[i];
  const p = maxScroll ? y / maxScroll : 0;
  await page.evaluate((y) => scrollTo({ top: y, behavior: "instant" }), y);
  await page.waitForTimeout(180);
  const settled = await settle();

  const state = await page.evaluate(() => {
    // A kinetic heading carries its real opacity on the split line units; the
    // engine forces the element itself to 1. Reading the element therefore
    // reports every kinetic headline as fully present, including on frames
    // where every one of its lines is at 0. Take the strongest line instead:
    // the heading is "peaked" when at least one unit has arrived.
    const cueOpacity = (el) => {
      const o = parseFloat(getComputedStyle(el).opacity) || 0;
      const units = el.querySelectorAll(".sc-split__i");
      if (!units.length) return o;
      let m = 0;
      units.forEach((u) => { m = Math.max(m, parseFloat(getComputedStyle(u).opacity) || 0); });
      return o * m;
    };
    const vis = [];
    // A worldflight's copy blocks are windowed against the whole track rather
    // than an act's progress, but they are the same thing to a reader: type that
    // has to arrive, hold, and leave. Grade them identically.
    document.querySelectorAll("[data-sc-cue],[data-sc-copy]").forEach((el) => {
      const o = cueOpacity(el);
      if (o <= 0.02) return;
      // On screen, not merely non-transparent. An element parked off-viewport
      // at opacity 1 is not a visible cue, and counting it produces phantom
      // findings that send you chasing a bug the reader never sees.
      const r = el.getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) return;
      vis.push({ t: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 46), o: +o.toFixed(2) });
    });
    const clips = [...document.querySelectorAll("video[data-sc-scrub]")].map((v) => ({
      // A continuous world legitimately keeps its clip chain outside the act
      // stack, driven by the page's own scroll value rather than by an act's
      // progress. Falling back to the clip's own class keeps that case
      // reporting instead of taking the whole run down before it writes
      // anything.
      painted: (v.closest("[data-sc-act]") ?? v).classList.contains("sc-has-clip"),
      t: +(v.currentTime || 0).toFixed(2),
      dur: +(v.duration || 0).toFixed(2),
      // How much of the viewport this clip's stage actually covers. A frozen
      // playhead only matters while the reader can see the stage.
      vis: (() => {
        const st = v.closest("[data-sc-stage]") || v.parentElement;
        if (!st) return 0;
        const b = st.getBoundingClientRect();
        return +(Math.max(0, Math.min(b.bottom, innerHeight) - Math.max(b.top, 0)) / innerHeight).toFixed(3);
      })(),
    }));
    // Rails and wipes move without changing any cue or clip time, so without
    // these a panning section reads as dead scroll.
    const rails = [...document.querySelectorAll("[data-sc-pan]")]
      .map((r) => Math.round(new DOMMatrixReadOnly(getComputedStyle(r).transform).m41));
    const wipes = [...document.querySelectorAll("[data-sc-reveal]")]
      .map((r) => getComputedStyle(r).clipPath);
    // Which act owns the middle of the viewport right now.
    let act = "-";
    document.querySelectorAll("[data-sc-act]").forEach((a) => {
      const r = a.getBoundingClientRect();
      if (r.top <= innerHeight / 2 && r.bottom >= innerHeight / 2) act = a.dataset.scAct;
    });
    // Where each pinned stage physically sits. Before an act reaches its pin
    // point the stage slides up the screen while its progress is still clamped
    // to 0, so the clip and cues are frozen and yet the view is very much
    // moving. Without this the run-up to every pinned act reads as dead scroll.
    const stages = [...document.querySelectorAll("[data-sc-stage]")]
      .map((s) => Math.round(s.getBoundingClientRect().top));
    // Worldflight legs. Opacity IS the crossfade, so it is state, not styling:
    // two samples with the same clip times but different leg opacities are a
    // dissolve in progress, not dead scroll.
    const segs = [...document.querySelectorAll("[data-sc-segment]")].map((el, i) => {
      const v = el.querySelector("video");
      return {
        i, label: el.dataset.scWaypoint || "",
        op: +(parseFloat(getComputedStyle(el).opacity) || 0).toFixed(3),
        painted: el.classList.contains("sc-has-clip"),
        t: v ? +(v.currentTime || 0).toFixed(3) : null,
      };
    });
    const world = document.querySelector('[data-sc-mode="worldflight"]');
    // Bespoke fixed stages can use flow markers for document travel while all
    // visible motion happens outside the engine's pin/scrub devices. Those
    // pages publish a compact representation of their actual visual state so
    // dead-scroll verification does not silently skip the whole experience.
    const customEls = [...document.querySelectorAll("[data-sc-verify-state]")];
    const custom = customEls.map((el) => el.getAttribute("data-sc-verify-state") || "");
    const customHold = customEls.some((el) => el.getAttribute("data-sc-verify-hold") === "true");
    return {
      cues: vis, clips, rails, wipes, act, stages, segs, custom, customHold,
      seg: world ? +(world.style.getPropertyValue("--sc-seg") || -1) : null,
      segp: world ? +(world.style.getPropertyValue("--sc-segp") || 0) : null,
      bg: getComputedStyle(document.documentElement).getPropertyValue("--sc-canvas").trim(),
    };
  });

  // Flat NN.png so ffmpeg can read the set as a numbered sequence for the
  // contact sheet. The scroll offset lives in report.json, not the filename.
  const name = `${String(i).padStart(2, "0")}.png`;
  await page.screenshot({ path: path.join(OUT, name) });

  // Contrast, measured on the COMPOSITED page rather than on the source media.
  // Sampling the video directly ignores every scrim, gradient and blend on top
  // of it, so a page can read as failing while looking fine, or the reverse.
  // Hide the text, shoot the same frame, hand the pixels back to the page, and
  // sample the real background under each line. Text over a scrubbing clip is
  // the one contrast case a static audit cannot cover: the frame beneath a
  // headline changes as you scroll, so it can pass on the poster and fail three
  // hundred pixels later. The direction is picked per line: light type fails on
  // the brightest patch, dark type on the darkest one.
  //
  // Fixed chrome is hidden along with the text. A fixed bar paints in FRONT of
  // whatever scrolls under it, so its own mark is not the background behind a
  // headline passing beneath it, and leaving it in reports a spurious failure
  // on an act that is fine.
  await page.evaluate(() => {
    document.querySelectorAll("body *").forEach((el) => {
      if (getComputedStyle(el).position !== "fixed") return;
      // A worldflight's stage and copy layer are fixed too, and they are the
      // exact opposite case: the stage IS the background behind every line, and
      // the copy layer carries the scrim that makes the line legible. Hiding
      // them samples the page ground instead of the film and reports the whole
      // page as failing while it looks fine.
      if (el.closest("[data-sc-world],[data-sc-world-copy]")) return;
      el.setAttribute("data-sc-shot-fixed", "");
    });
  });
  await page.addStyleTag({
    content: "[data-sc-cue],[data-sc-cue] *,[data-sc-copy],[data-sc-copy] *," +
             "[data-sc-shot-fixed]{visibility:hidden!important}",
  });
  const bare = (await page.screenshot({ type: "jpeg", quality: 80 })).toString("base64");
  const contrast = await page.evaluate(async ({ b64, dpr }) => {
    const img = new Image();
    img.src = "data:image/jpeg;base64," + b64;
    await img.decode();
    const c = document.createElement("canvas");
    const g = c.getContext("2d", { willReadFrequently: true });
    const lum = (r, gr, b) => {
      const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
      return 0.2126 * f(r) + 0.7152 * f(gr) + 0.0722 * f(b);
    };
    const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    const cueOpacity = (el) => {
      const o = parseFloat(getComputedStyle(el).opacity) || 0;
      const units = el.querySelectorAll(".sc-split__i");
      if (!units.length) return o;
      let m = 0;
      units.forEach((u) => { m = Math.max(m, parseFloat(getComputedStyle(u).opacity) || 0); });
      return o * m;
    };
    const out = [];
    document.querySelectorAll("[data-sc-cue],[data-sc-copy]").forEach((el) => {
      if (cueOpacity(el) < 0.85) return;
      if (!(el.textContent || "").trim()) return;
      const r = el.getBoundingClientRect();
      if (r.width < 8 || r.height < 8 || r.bottom < 0 || r.top > innerHeight) return;
      // Clamp the sampled rect to the viewport. The part of a pinned act's copy
      // that has scrolled above the fold is not on screen, so whatever sits in
      // those pixels is not the background behind anything the reader can see.
      const vl = Math.max(0, r.left), vt = Math.max(0, r.top);
      const vr = Math.min(innerWidth, r.right), vb = Math.min(innerHeight, r.bottom);
      if (vr - vl < 8 || vb - vt < 8) return;
      const x = vl * dpr, y2 = vt * dpr;
      const w = Math.min((vr - vl) * dpr, img.width - x), h = Math.min((vb - vt) * dpr, img.height - y2);
      if (w < 2 || h < 2) return;
      c.width = 32; c.height = 16;
      g.drawImage(img, x, y2, w, h, 0, 0, 32, 16);
      const d = g.getImageData(0, 0, 32, 16).data;
      let maxL = 0, minL = 1, sum = 0, n = 0;
      for (let k = 0; k < d.length; k += 4) {
        const L = lum(d[k], d[k + 1], d[k + 2]);
        if (L > maxL) maxL = L;
        if (L < minL) minL = L;
        sum += L; n++;
      }
      const cs = getComputedStyle(el);
      const fg = cs.color.match(/[\d.]+/g).map(Number);
      const fl = lum(fg[0], fg[1], fg[2]);
      // An element that paints its own opaque background (a button, a chip) is
      // an ordinary static contrast case: grade its text against that fill, not
      // against whatever the page happens to show behind it. Hiding the element
      // to sample the backdrop necessarily hides its background too, so without
      // this every solid CTA reports a spurious failure.
      const bg = (cs.backgroundColor.match(/[\d.]+/g) || []).map(Number);
      const opaqueBg = bg.length >= 3 && (bg.length < 4 || bg[3] > 0.5);
      if (opaqueBg) {
        const bl = lum(bg[0], bg[1], bg[2]);
        out.push({
          t: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40),
          dir: "own-fill",
          worst: +ratio(fl, bl).toFixed(2), mean: +ratio(fl, bl).toFixed(2),
        });
        return;
      }
      // Pick the direction from the foreground. Light type on a dark page fails
      // on the brightest patch under it; dark type on a light page (a high-key
      // world, ink over media) fails on the DARKEST patch, and grading that
      // against maxL is the most lenient reading available, so a page can report
      // clean over text that is failing. Compare the ink to the mean background
      // and grade against whichever extreme is on the ink's own side.
      const meanL = sum / n;
      const dark = fl < meanL;
      out.push({
        t: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40),
        dir: dark ? "dark-on-light" : "light-on-dark",
        worst: +ratio(fl, dark ? minL : maxL).toFixed(2),
        mean: +ratio(fl, meanL).toFixed(2),
      });
    });
    return out;
  }, { b64: bare, dpr: 2 });
  await page.evaluate(() => {
    const t = [...document.querySelectorAll("style")].pop();
    if (t && t.textContent.includes("data-sc-cue")) t.remove();
    document.querySelectorAll("[data-sc-shot-fixed]").forEach((el) => el.removeAttribute("data-sc-shot-fixed"));
  });

  report.push({ i, y, pct: +(p * 100).toFixed(0), settled, contrast, ...state });
  if (WORLD) {
    console.log(`  ${name}  settled=${settled}  copy=${state.cues.length}  leg=${state.seg}@${state.segp}  ` +
      `legs=${state.segs.map((g) => `${g.op > 0.002 ? (g.painted ? g.t : "poster") : "-"}${g.op > 0.002 && g.op < 0.998 ? "*" + g.op : ""}`).join(",")}`);
  } else {
    console.log(`  ${name}  settled=${settled}  cues=${state.cues.length}  clips=${state.clips.map((c) => (c.painted ? c.t : "poster")).join(",")}`);
  }
}

fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify({ doc, report, consoleErrors, failed }, null, 2));

if (consoleErrors.length) console.log("\nCONSOLE ERRORS:\n  " + consoleErrors.join("\n  "));
if (failed.length) console.log("\nFAILED REQUESTS:\n  " + failed.join("\n  "));

// Dead-scroll detector: consecutive positions where nothing visibly changed.
// This is the failure the eye misses and the reason to shoot contiguously.
// Opacity counts as change: a cue mid-fade is motion, and comparing only which
// cues exist would call a crossfade dead.
const sig = (s) => JSON.stringify([
  s.cues.map((c) => c.t + ":" + c.o),
  s.clips.map((c) => c.t),
  s.rails,
  s.wipes,
  s.stages,
  s.custom || [],
]);
// Only inside pinned acts. A flow section or a footer that holds still across
// two sample positions is a page behaving correctly, not dead scroll, and
// flagging it trains you to ignore the signal.
const PINNED = new Set(["scrub", "pin", "pan"]);
const dead = [];
if (WORLD) {
  // Every pixel of a worldflight track is pinned by construction, so there is
  // no "correctly still" region to exclude and the whole page is fair game.
  // Three independent things can carry the motion: the film advancing, a leg
  // dissolving into the next, and a copy window opening or closing. Dead scroll
  // is all three holding at once.
  const wsig = (s) => JSON.stringify([
    s.segs.map((g) => g.t),
    s.segs.map((g) => g.op),
    s.cues.map((c) => c.t + ":" + c.o),
  ]);
  // Not under reduced motion. There the film is deliberately never fetched, so
  // the middle of a leg holds a single still frame and every pair of samples in
  // it is identical BY DESIGN. Flagging that reports the accessibility path as
  // broken every single run, which is how a real finding gets ignored. What
  // matters here is whether the story still reads, and the copy-window and
  // contrast passes below answer that.
  for (let i = 1; !REDUCED && i < report.length; i++) {
    const a = report[i - 1], b = report[i];
    // Tighter than the act gate: a leg is about one viewport of scroll, so a
    // quarter-viewport window would only ever compare four points per leg.
    if (b.y - a.y < doc.vh * 0.12) continue;
    if (wsig(a) === wsig(b)) {
      dead.push(`${a.pct}% -> ${b.pct}% (leg ${a.seg} > ${b.seg})`);
    }
  }
} else {
  for (let i = 1; i < report.length; i++) {
    const a = report[i - 1], b = report[i];
    const hasCustomState = (a.custom?.length || 0) > 0 || (b.custom?.length || 0) > 0;
    if (!PINNED.has(a.act) && !PINNED.has(b.act) && !hasCustomState) continue;
    // A page may explicitly declare an authored hold, such as a resolved close
    // or the stable accessibility frame under reduced motion. It has to be
    // declared by the visible stage; ordinary flow content stays exempt as it
    // was before this custom-state path existed.
    if (a.customHold && b.customHold) continue;
    // Two samples a few dozen pixels apart SHOULD look the same. Only flag a gap
    // wide enough that a reader would notice nothing happening in it.
    if (b.y - a.y < doc.vh * 0.25) continue;
    if (sig(a) === sig(b)) dead.push(`${a.pct}% -> ${b.pct}% (${a.act} > ${b.act})`);
  }
}
console.log(dead.length ? `\nDEAD SCROLL between: ${dead.join(", ")}`
  : WORLD && REDUCED ? "\ndead-scroll check skipped: reduced motion holds each leg on one still frame by design"
  : "\nno dead scroll detected");

// FROZEN CLIP. The reader is scrolling, a scrub stage is on screen, and its
// playhead is not moving: a still photograph sliding up the page. Dead scroll
// cannot see this, because the stage IS moving, which is the whole problem.
//
// A hold on the first or last frame is always a defect. A hold in the middle
// can be an intentional `dwell` settle, so it only counts once it outlasts one.
// Skipped under reduced motion, where no clip is ever fetched on purpose.
if (!REDUCED) {
  const nClips = report[0]?.clips?.length || 0;
  const VIS = 0.55, EPS = 0.012, MIN = doc.vh * 0.15;
  const frozen = [];
  for (let c = 0; c < nClips; c++) {
    let run = null;
    const flush = () => {
      if (!run) return;
      const kind = run.t < 0.05 ? "entry" : (run.dur && run.t > run.dur - 0.08 ? "exit" : "mid");
      const need = kind === "mid" ? doc.vh * 0.5 : MIN;
      if (run.to - run.from >= need) frozen.push({ c, kind, ...run });
      run = null;
    };
    for (let i = 1; i < report.length; i++) {
      const a = report[i - 1].clips?.[c], b = report[i].clips?.[c];
      if (!a || !b) { flush(); continue; }
      const seen = a.vis >= VIS && b.vis >= VIS;
      const stuck = Math.abs(b.t - a.t) < EPS;
      if (seen && stuck && b.painted) {
        if (!run) run = { from: report[i - 1].y, to: report[i].y, t: b.t, dur: b.dur };
        else run.to = report[i].y;
      } else flush();
    }
    flush();
  }
  if (frozen.length) {
    console.log("\nFROZEN CLIP (still image while the page moves):\n  " + frozen.map((f) => {
      const px = f.to - f.from;
      const where = f.kind === "entry" ? "held on its FIRST frame while the stage slides in"
        : f.kind === "exit" ? "held on its LAST frame while the stage slides out"
        : `held mid-clip at ${f.t.toFixed(2)}s, longer than a dwell settle`;
      return `clip ${f.c}: ${px}px (${(px / doc.vh).toFixed(2)} viewports) ${where}`;
    }).join("\n  ") + "\n  Fix: let the clip map across the stage's whole visible life. That is the\n  engine default; data-sc-clip-map=\"travel\" turns it off. See devices.md.");
  } else if (nClips) {
    console.log(`all ${nClips} scrub clip(s) keep moving whenever they are on screen`);
  }
}

// Worldflight findings. A leg that never reaches full opacity is a weight or a
// seam that is wrong: the reader is shown a permanent dissolve between two
// clips and never the leg itself. A leg stuck on its poster is a clip that
// never loaded or never decoded, and it passes every other check on this page
// because a poster looks exactly like a paused film.
if (WORLD) {
  const segPeak = {};
  report.forEach((s) => (s.segs || []).forEach((g) => {
    const k = `${g.i}${g.label ? ' "' + g.label + '"' : ""}`;
    segPeak[k] = segPeak[k] || { op: 0, painted: false, hasClip: g.t !== null };
    segPeak[k].op = Math.max(segPeak[k].op, g.op);
    segPeak[k].painted = segPeak[k].painted || g.painted;
  }));
  const faint = Object.entries(segPeak).filter(([, v]) => v.op < 0.99);
  // Under reduced motion no clip is ever fetched, on purpose. Every leg is
  // legitimately on its poster, and reporting that as a fault buries the one
  // finding this pass exists for: whether the story still reads without motion.
  const posters = REDUCED ? [] : Object.entries(segPeak).filter(([, v]) => v.hasClip && !v.painted);
  if (faint.length) console.log("\nLEGS THAT NEVER REACH FULL OPACITY:\n  " +
    faint.map(([k, v]) => `${v.op.toFixed(2)} leg ${k}`).join("\n  "));
  if (posters.length) console.log("\nLEGS STUCK ON POSTER (clip never painted):\n  " +
    posters.map(([k]) => `leg ${k}`).join("\n  "));
  if (!faint.length && !posters.length)
    console.log(`all ${Object.keys(segPeak).length} legs reach full opacity` +
      (REDUCED ? " (posters only, as reduced motion requires)" : " and paint a real frame"));
}

// Cues that never reach full strength anywhere on the page. A headline peaking
// at 0.6 is a mis-set cue window, and it is invisible as a bug because the
// element IS there, just never quite arriving.
const peak = {};
report.forEach((s) => s.cues.forEach((c) => { peak[c.t] = Math.max(peak[c.t] || 0, c.o); }));
const weak = Object.entries(peak).filter(([, o]) => o < 0.8);
if (weak.length) console.log("\nCUES THAT NEVER PEAK:\n  " + weak.map(([t, o]) => `${o} "${t}"`).join("\n  "));

// Contrast over media, graded at the worst frame each line is ever shown on.
const worstBy = {};
report.forEach((s) => (s.contrast || []).forEach((c) => {
  if (!worstBy[c.t] || c.worst < worstBy[c.t].worst) worstBy[c.t] = c;
}));
const fails = Object.values(worstBy).filter((c) => c.worst < 3);
const thin = Object.values(worstBy).filter((c) => c.worst >= 3 && c.worst < 4.5);
if (fails.length) console.log("\nCONTRAST FAIL (worst frame < 3:1):\n  " +
  fails.map((c) => `${c.worst}:1 (mean ${c.mean}) "${c.t}"`).join("\n  "));
if (thin.length) console.log("\nCONTRAST THIN (3:1 to 4.5:1, ok for large display type only):\n  " +
  thin.map((c) => `${c.worst}:1 "${c.t}"`).join("\n  "));
if (!fails.length && !thin.length && Object.keys(worstBy).length)
  console.log("\ncontrast over media: all cues clear 4.5:1 at their worst frame");

await browser.close();

// Contact sheet. The point of shooting contiguously is to look at the frames
// side by side; a folder of 20 PNGs does not get looked at that way.
const FFMPEG = [
  process.env.SCROLLCRAFT_FFMPEG,
  ...(fs.existsSync(path.join(process.env.HOME || "", "AppData/Local/Microsoft/WinGet/Packages"))
    ? fs.readdirSync(path.join(process.env.HOME, "AppData/Local/Microsoft/WinGet/Packages"))
        .filter((d) => d.startsWith("Gyan.FFmpeg"))
        .flatMap((d) => {
          const base = path.join(process.env.HOME, "AppData/Local/Microsoft/WinGet/Packages", d);
          return fs.readdirSync(base).map((v) => path.join(base, v, "bin/ffmpeg.exe"));
        })
    : []),
  "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "ffmpeg",
].find((p) => p && (p === "ffmpeg" || fs.existsSync(p)));

if (FFMPEG) {
  const cols = Math.min(5, report.length);
  const rows = Math.ceil(report.length / cols);
  const { spawnSync } = await import("node:child_process");
  const r = spawnSync(FFMPEG, [
    "-y", "-v", "error", "-i", path.join(OUT, "%02d.png"),
    "-vf", `scale=520:-1,tile=${cols}x${rows}`, "-frames:v", "1",
    path.join(OUT, "sheet.png"),
  ]);
  if (r.status === 0) console.log(`contact sheet: ${path.join(OUT, "sheet.png")}`);
  else console.log("contact sheet skipped (needs a full ffmpeg build; scale/tile are missing from stripped ones)");
}

console.log(`\nshots + report.json in ${OUT}`);
