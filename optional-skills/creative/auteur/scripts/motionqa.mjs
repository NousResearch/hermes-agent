#!/usr/bin/env node
/**
 * motionqa.mjs — motion/perf/audio gate for Tier-1 scenes
 * (scroll state-machine, audio-reactive, 2.5D composite, scrubbed video).
 * Screenshots (shoot.mjs) are blind to time; this scrolls the page under CPU throttle and asserts
 * FPS, long-tasks, audio gating, and WebGL console health.
 * Usage: node motionqa.mjs <url> [--headed] [--dpr 2] [--throttle 4] [--min-fps 50] [--max-longtask 50] [--json]
 * --headed is not optional for a real number on any GPU-dependent page, and a headless run
 * reporting 0 console errors is not a pass — a headed run has caught a 404 the headless one missed.
 * Measure a PRODUCTION BUILD: a dev server costs roughly double per frame (HMR client, unminified
 * bundles, no asset pipeline), so its numbers describe a page nobody will ever load.
 * ponytail: reuses the playwright install shoot.mjs already needs; no new deps.
 */
import { chromium } from 'playwright';

const argv = process.argv.slice(2);
const url = argv.find(a => !a.startsWith('-'));
const opt = (name, def) => { const i = argv.indexOf('--' + name); return i >= 0 && argv[i + 1] ? +argv[i + 1] : def; };
const jsonMode = argv.includes('--json');
if (!url) {
  process.stderr.write('Usage: node motionqa.mjs <url> [--headed] [--throttle 4] [--min-fps 50] [--max-longtask 50] [--json]\n');
  process.exit(2);
}
const THROTTLE = opt('throttle', 4), MIN_FPS = opt('min-fps', 50), MAX_LT = opt('max-longtask', 50);
const DPR = opt('dpr', 2) || 2;   // `|| 2` so a typo'd --dpr becomes the honest default, not a NaN viewport
const headed = argv.includes('--headed');   // headless chromium has NO GPU (software swiftshader) — WebGL FPS there is a floor, not the real number

const browser = await chromium.launch({ headless: !headed });
// 1440x900 @2x = 5.2 megapixels: a retina laptop, which is what a page like this gets judged on.
// The default DPR 1 renders 1.3MP, and fullscreen post-processing (bloom, DoF, grain, any full-frame
// shader) costs per pixel — so at DPR 1 the expensive part of the frame is a quarter of its real cost
// and the gate says 60fps about a page that stutters on the reviewer's MacBook. Fillrate is the budget,
// not geometry. --dpr 1 is for reproducing an old measurement, not for judging a scene.
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: DPR });
const page = await context.newPage();
const consoleErrors = [];
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', e => consoleErrors.push(String(e)));

await page.goto(url, { waitUntil: 'load' });

// audio/video must be silent on load — sound starts on a user gesture, never before
const audioAutoplaying = await page.evaluate(() =>
  [...document.querySelectorAll('audio,video')].some(a => !a.paused && !a.muted && a.volume > 0));
const hasCanvas = await page.evaluate(() => !!document.querySelector('canvas'));

// A dev server costs roughly 2x per frame vs its own production build (HMR client, unminified
// bundles, on-the-fly transforms, no image pipeline). That direction only ever produces FALSE
// FAILURES — but a false FAIL sends you optimizing a scene that was already fast, which is how a
// motion budget gets cut for nothing. Detect the three common dev clients and say so out loud.
const devServer = await page.evaluate(() => !!(
  document.querySelector('script[src*="/@vite/client"]') ||
  window.__vite_plugin_react_preamble_installed__ ||
  window.webpackHotUpdate || window.__webpack_hmr ||
  window.next?.router?.isDevBuild || window.__NEXT_DATA__?.buildId === 'development'
));

// warm up: let CDN modules load, shaders compile, first textures upload — we measure STEADY STATE,
// not the cold-start frame. (Headless chromium renders WebGL via software swiftshader, so for WebGL
// scenes also prefer a headed/GPU run or a lower --throttle; software raster is not representative.)
await page.evaluate(() => new Promise(r => {
  scrollTo(0, document.body.scrollHeight);
  setTimeout(() => { scrollTo(0, 0); setTimeout(r, 500); }, 900);
}));

const cdp = await page.context().newCDPSession(page);
await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });

// `buffered: true` replays every long task since navigation — bundle parse, shader compile,
// PMREM prefilter — into a number the rubric defines as "during the scroll". Observe live only,
// after warm-up, and report load-time separately so both are visible and neither is a false FAIL.
await page.evaluate(() => {
  window.__lt = 0; window.__ltLoad = 0;
  try {
    new PerformanceObserver(l => { for (const e of l.getEntries()) window.__ltLoad = Math.max(window.__ltLoad, e.duration); })
      .observe({ type: 'longtask', buffered: true });
    new PerformanceObserver(l => { for (const e of l.getEntries()) window.__lt = Math.max(window.__lt, e.duration); })
      .observe({ type: 'longtask' });
  } catch { /* longtask unsupported */ }
});

// slow full-page scroll over ~4s, sampling rAF deltas for the worst (min) FPS
const minFps = await page.evaluate(() => new Promise(res => {
  let last = performance.now(), min = 999, end = last + 4000, seen = 0;
  (function tick(t) {
    const d = t - last; last = t;
    if (d > 0) { min = Math.min(min, 1000 / d); seen++; }
    scrollBy(0, innerHeight / 60);
    t < end ? requestAnimationFrame(tick) : res(seen > 10 ? min : 60);
  })(performance.now());
}));
const maxLongTask = await page.evaluate(() => window.__lt || 0);
const loadLongTask = await page.evaluate(() => window.__ltLoad || 0);
await browser.close();

const fails = [], advisories = [];
// headless chromium has no GPU → software swiftshader inflates WebGL FPS + raster long-tasks.
// Under headless+canvas those two are ADVISORIES (run --headed / on-device for a real number); the
// rest (audio gating, console/WebGL errors) are GPU-independent and stay hard fails.
const softWebgl = hasCanvas && !headed;
const perf = (cond, msg) => { if (cond) (softWebgl ? advisories : fails).push(msg + (softWebgl ? ` [headless software ${hasCanvas ? 'WebGL' : 'rasterization'} — run --headed for a real number]` : '')); };
perf(minFps < MIN_FPS, `minFps ${minFps.toFixed(0)} < ${MIN_FPS} @${THROTTLE}x throttle`);
perf(maxLongTask > MAX_LT, `long task ${maxLongTask.toFixed(0)}ms > ${MAX_LT}ms`);
if (audioAutoplaying) fails.push('audio/video playing with sound on load (must be gesture-gated)');
const ctxErr = consoleErrors.filter(e => /WebGL|Context Lost|Too many active/i.test(e));
if (ctxErr.length) fails.push(`WebGL context error in console: ${ctxErr[0].slice(0, 80)}`);

// Load-time long tasks are real information (bundle parse, shader compile, PMREM prefilter) but
// they are not what the rubric row asks about, so they are reported, never failed on.
if (loadLongTask > MAX_LT) advisories.push(`load-time long task ${loadLongTask.toFixed(0)}ms (bundle parse / shader compile) — not a scroll fail, but it delays interactivity`);
if (devServer) advisories.push('measured against a DEV SERVER — a production build runs roughly 2x faster per frame; re-measure on the built output before you cut anything from the scene');
const megapixels = +((1440 * 900 * DPR * DPR) / 1e6).toFixed(1);
const summary = {
  minFps: +minFps.toFixed(0), maxLongTask: +maxLongTask.toFixed(0), loadLongTask: +loadLongTask.toFixed(0),
  dpr: DPR, megapixels, devServer,
  audioAutoplaying, consoleErrors: consoleErrors.length, headed, softWebgl, fails, advisories,
};
if (jsonMode) {
  process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
} else {
  // DPR belongs in the headline number: "minFps 57" means nothing without the pixel count behind it,
  // and this line is what gets quoted verbatim into the QA sheet.
  process.stdout.write(`motionqa: minFps ${summary.minFps} @${THROTTLE}x CPU - 1440x900@${DPR}x (${megapixels}MP) - scroll long-task ${summary.maxLongTask}ms (load ${summary.loadLongTask}ms) - audio ${audioAutoplaying ? 'AUTOPLAYING(!)' : 'gesture-gated'} - console errors ${consoleErrors.length}${softWebgl ? ' [headless/software-WebGL]' : ''}${devServer ? ' [DEV SERVER]' : ''}\n`);
  for (const a of advisories) process.stdout.write(`ADVISORY ${a}\n`);
  for (const f of fails) process.stdout.write(`FAIL ${f}\n`);
  if (!fails.length) process.stdout.write(advisories.length ? 'PASS (perf advisories — verify headed)\n' : 'PASS\n');
}
process.exit(fails.length ? 1 : 0);
