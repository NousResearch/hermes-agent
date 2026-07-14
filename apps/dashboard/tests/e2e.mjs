// End-to-end smoke test for Hermes Hub.
//
// Drives a real Chromium against a running server (offline mode is fine) and
// exercises every widget plus the viewer, palette, edit mode and theming.
//
// Usage:
//   node apps/dashboard/tests/e2e.mjs [baseURL] [screenshotDir]
// Requires `playwright-core` resolvable and a Chromium binary — pass its path
// via PLAYWRIGHT_CHROMIUM (defaults to /opt/pw-browsers/chromium).

import { pathToFileURL } from "node:url";

// Resolve playwright-core from the usual node_modules, or from PW_CORE_DIR
// (a directory containing node_modules/playwright-core) when the repo itself
// doesn't ship node dependencies.
const chromium = await (async () => {
  try {
    return (await import("playwright-core")).chromium;
  } catch {
    const dir = process.env.PW_CORE_DIR;
    if (!dir) throw new Error("playwright-core not found; set PW_CORE_DIR");
    const mod = await import(pathToFileURL(`${dir}/node_modules/playwright-core/index.mjs`));
    return mod.chromium;
  }
})();

const BASE = process.argv[2] || "http://127.0.0.1:8787";
const SHOT_DIR = process.argv[3] || "";
const EXECUTABLE = process.env.PLAYWRIGHT_CHROMIUM || "/opt/pw-browsers/chromium";

let failures = 0;
const check = (name, condition) => {
  console.log(`${condition ? "  ✓" : "  ✗ FAIL"} ${name}`);
  if (!condition) failures++;
};

const shot = async (page, name) => {
  if (SHOT_DIR) await page.screenshot({ path: `${SHOT_DIR}/${name}.png`, fullPage: name.includes("full") });
};

const browser = await chromium.launch({ executablePath: EXECUTABLE });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on("pageerror", (err) => errors.push(String(err)));
page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });
if (process.env.E2E_TRACE_SYNC) {
  page.on("response", (r) => {
    if (r.url().includes("/api/state")) {
      console.log(`      [state ${new Date().toISOString().slice(17, 23)}] ${r.request().method()} -> ${r.status()}`);
    }
  });
}

console.log(`— loading ${BASE}`);

// Sync-liveness probe (enabled with E2E_TRACE_SYNC): mutate state, then see
// whether the server revision advances within the debounce window.
async function syncProbe(label) {
  if (!process.env.E2E_SYNC_PROBES) return;
  const before = await page.evaluate(async () => (await (await fetch("/api/state")).json()).rev);
  await page.evaluate((l) => {
    // schedule a raw store mutation through the page's own module graph
    return import("/js/store.js").then(({ store }) => {
      store.update((s) => { s.search.engine = s.search.engine; }, "probe-" + l);
    });
  }, label);
  await page.waitForTimeout(2500);
  const after = await page.evaluate(async () => (await (await fetch("/api/state")).json()).rev);
  console.log(`    [probe ${label}] server rev ${before} -> ${after}  ${after > before ? "SYNC ALIVE" : "SYNC DEAD"}`);
}

await page.goto(BASE, { waitUntil: "networkidle" });

// Make the run idempotent: reset local state to defaults and let sync push
// the reset to the server (previous runs' data would otherwise be adopted).
await page.evaluate(() => import("/js/store.js").then(({ store }) => store.reset()));
await page.waitForTimeout(1600); // debounced sync push of the reset
await page.reload({ waitUntil: "networkidle" });

// ---- shell -----------------------------------------------------------------
check("title", (await page.title()) === "Hermes Hub");
check("topbar brand", await page.locator(".brand-name").innerText() === "HERMES//HUB");
check("dark theme default", await page.evaluate(() => document.documentElement.dataset.theme) === "dark");

// ---- widgets render --------------------------------------------------------
for (const type of ["clock", "worldstate", "agent", "weather", "launcher", "news", "tasks", "markets", "calendar", "notes"]) {
  await page.waitForSelector(`.widget-${type}`, { timeout: 10000 });
  check(`widget ${type} present`, true);
}
await page.waitForSelector(".news-item", { timeout: 10000 });
check("news items rendered", (await page.locator(".news-item").count()) > 3);
check("clock shows time", /\d{1,2}:\d{2}/.test(await page.locator(".clock-time").innerText()));
check("weather temp shown", /-?\d+°/.test(await page.locator(".weather-temp").innerText()));
check("market rows", (await page.locator(".market-row").count()) >= 3);
check("worldstate domains", (await page.locator(".ws-row").count()) >= 5);
check("worldstate levels are labeled", (await page.locator(".ws-row .level-chip").first().innerText()).length > 2);
check("sample badges when offline", (await page.locator(".widget-badge:not([hidden])").count()) >= 2);
await shot(page, "01-dashboard-dark-full");
await syncProbe("P1-after-boot");

// ---- state of the world interactions ---------------------------------------
const wsRow = page.locator(".ws-row").nth(1);
await wsRow.click();
await page.waitForSelector(".ws-detail", { timeout: 5000 });
check("worldstate explanation expands", (await page.locator(".ws-explanation").first().innerText()).length > 20);
await page.waitForSelector(".ws-signal", { timeout: 5000 });
await page.locator(".ws-signal").first().click();
await page.waitForSelector(".viewer", { timeout: 5000 });
check("signal opens in-app viewer", true);
await shot(page, "02-viewer-reader");
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// ---- news opens in-app ------------------------------------------------------
await page.locator(".news-item").first().click();
await page.waitForSelector(".viewer", { timeout: 5000 });
check("news opens in-app viewer", true);
check("viewer has reader/embed tabs", (await page.locator(".viewer-tab").count()) === 2);
await page.locator(".viewer-tab", { hasText: "EMBED" }).click();
check("embed tab shows iframe", (await page.locator(".viewer-frame").count()) === 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// ---- launcher opens in-app --------------------------------------------------
await page.locator(".app-tile").first().click();
await page.waitForSelector(".viewer", { timeout: 5000 });
check("app tile opens in-app viewer (embed)", (await page.locator(".viewer-frame").count()) === 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// ---- news topic tabs ---------------------------------------------------------
await page.locator(".tab", { hasText: "Science" }).click();
await page.waitForFunction(() =>
  document.querySelector('.tab[aria-selected="true"]')?.textContent === "Science");
await page.waitForSelector(".news-item");
check("topic switch renders items", (await page.locator(".news-item").count()) > 2);

// ---- tasks -------------------------------------------------------------------
const taskInput = page.locator(".task-form .input");
await taskInput.fill("E2E: buy coffee beans");
await taskInput.press("Enter");
check("task added", await page.locator(".task-text", { hasText: "E2E: buy coffee beans" }).count() === 1);
await page.locator(".task-item", { hasText: "E2E: buy coffee beans" }).locator("input[type=checkbox]").check();
check("task toggles done", (await page.locator(".task-done", { hasText: "E2E: buy coffee beans" }).count()) === 1);

// ---- notes autosave ----------------------------------------------------------
await page.locator(".note-pad").fill("E2E note line one\nsecond line");
await page.waitForTimeout(700); // debounce
const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("hermesHub.v1")).notes.items);
check("note persisted to localStorage", saved.some((n) => n.text.startsWith("E2E note")));

// ---- calendar ----------------------------------------------------------------
await page.locator(".cal-form .input").fill("E2E standup");
await page.locator(".cal-form .btn-primary").click();
check("calendar event added", (await page.locator(".cal-event-title", { hasText: "E2E standup" }).count()) === 1);
check("calendar dot on selected day", (await page.locator(".cal-cell .cal-dot").count()) >= 1);
await syncProbe("P2-after-widgets");

// ---- agent: chat drives dashboard actions -----------------------------------
await page.waitForFunction(() =>
  document.querySelector(".agent-mode")?.textContent !== "…");
check("agent mode chip resolved", ["LOCAL MODE", "OFFLINE"].includes(await page.locator(".agent-mode").innerText()) ||
  (await page.locator(".agent-mode").innerText()).startsWith("CLAUDE"));

await page.locator(".agent-input").fill("add task e2e agent made this to errands");
await page.locator(".agent-form .btn-primary").click();
await page.waitForSelector(".agent-chip", { timeout: 10000 });
check("agent executed add_task action", (await page.locator(".agent-chip").first().innerText()).includes("added task"));
// tasks widget should have switched to the new list and show the task
await page.waitForFunction(() =>
  [...document.querySelectorAll(".task-text")].some((el) => el.textContent.includes("e2e agent made this")));
check("agent task visible in Lists widget", true);

await page.locator(".agent-input").fill("open GitHub");
await page.locator(".agent-form .btn-primary").click();
await page.waitForSelector(".viewer", { timeout: 10000 });
check("agent opens app in viewer", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// briefing
await page.locator(".agent-quick .link-btn").first().click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => el.textContent.includes("FOCUS")), null, { timeout: 15000 });
check("briefing produced with FOCUS section", true);
check("briefing recognizes the agent-created task", await page.evaluate(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => el.textContent.includes("e2e agent made this"))));
await shot(page, "05-agent");

// ---- summarize buttons -------------------------------------------------------
check("widget summarize buttons present", (await page.locator(".widget-controls .sum-btn").count()) >= 6);
await page.locator(".news-item .sum-inline").first().click();
await page.waitForSelector(".sum-pop", { timeout: 10000 });
await page.waitForFunction(() => document.querySelector(".sum-body p"));
check("news item summary shows text", (await page.locator(".sum-body").innerText()).length > 20);
check("summary labeled with engine", ["LOCAL", "CLAUDE"].includes(await page.locator(".sum-mode").innerText()));
await shot(page, "06-summary");
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// widget-level summarize (worldstate)
await page.locator(".widget-worldstate .widget-controls .sum-btn").click();
await page.waitForSelector(".sum-pop", { timeout: 10000 });
check("worldstate widget summary opens", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// ---- search bar --------------------------------------------------------------
// Stub window.open — the sandbox has no outbound network, so a real popup
// would never commit its navigation. We assert on the URL the app requested.
await page.evaluate(() => { window.__opened = []; window.open = (url) => { window.__opened.push(url); return null; }; });
await page.locator(".search-input").fill("hello world");
await page.locator(".search-input").press("Enter");
const opened = await page.evaluate(() => window.__opened);
check("search targets engine", opened.length === 1 && opened[0].includes("google.com/search?q=hello%20world"));
await page.locator(".search-input").fill("yt lofi beats");
await page.locator(".search-input").press("Enter");
const opened2 = await page.evaluate(() => window.__opened);
check("bang prefix switches engine", opened2[1]?.includes("youtube.com/results"));

// ---- command palette -----------------------------------------------------------
await page.keyboard.press("Control+k");
await page.waitForSelector(".palette", { timeout: 5000 });
await page.locator(".palette-input").fill("github");
check("palette filters commands", (await page.locator(".palette-item").count()) >= 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".palette", { state: "detached" });

// ---- edit mode: add/resize/remove/drag persistence -------------------------------
await page.locator("#edit-toggle").click();
await page.waitForSelector(".add-gallery");
const widgetCountBefore = await page.locator(".widget:not(.add-gallery)").count();
await page.locator(".gallery-item", { hasText: "Clock" }).click();
check("gallery adds widget", (await page.locator(".widget:not(.add-gallery)").count()) === widgetCountBefore + 1);
await page.locator(".widget-remove").last().click();
check("remove widget works", (await page.locator(".widget:not(.add-gallery)").count()) === widgetCountBefore);
await shot(page, "03-edit-mode");
await page.locator("#edit-toggle").click();

// ---- theme cycle + light mode screenshot -----------------------------------------
const themeBtn = page.locator(".topbar-actions .btn").nth(1);
await themeBtn.click(); // dark → auto
await themeBtn.click(); // auto → light
check("light theme applies", await page.evaluate(() => document.documentElement.dataset.theme) === "light");
await page.waitForTimeout(200);
await shot(page, "04-dashboard-light-full");
await themeBtn.click(); // light → dark
check("back to dark", await page.evaluate(() => document.documentElement.dataset.theme) === "dark");
await syncProbe("P3-pre-reload");

// ---- persistence across reload ----------------------------------------------------
await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector(".task-text");
const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem("hermesHub.v1")).tasks.lists);
check("tasks survive reload",
  persisted.some((l) => l.items.some((i) => i.text === "E2E: buy coffee beans")) &&
  persisted.some((l) => l.name === "Errands" && l.items.some((i) => i.text.includes("e2e agent made this"))));
check("theme survives reload", await page.evaluate(() => document.documentElement.dataset.theme) === "dark");

// ---- PWA -------------------------------------------------------------------------
const manifestOk = await page.evaluate(async () => {
  const res = await fetch("/manifest.webmanifest");
  if (!res.ok) return false;
  const mf = await res.json();
  return mf.name === "HERMES//HUB" && mf.icons.length >= 2;
});
check("PWA manifest served and valid", manifestOk);
const iconOk = await page.evaluate(async () => (await fetch("/icons/icon-192.png")).ok);
check("PWA icon served", iconOk);
const swOk = await Promise.race([
  page.evaluate(() => navigator.serviceWorker.ready.then(() => true)),
  new Promise((resolve) => setTimeout(() => resolve(false), 8000)),
]);
check("service worker registered", swOk === true);

// ---- sync: a second "device" pulls the same state --------------------------------
// Nudge the store so a fresh push fires (a reload can cancel a pending
// debounced push), then wait until the server copy has this run's task.
console.log("    [debug] pre-nudge server rev:", await page.evaluate(async () =>
  (await fetch("/api/state")).json().then((d) => `${d.rev} hasCoffee=${JSON.stringify(d.state || {}).includes("coffee beans")}`)));
console.log("    [debug] page A syncRev:", await page.evaluate(() => localStorage.getItem("hermesHub.syncRev")));
await page.locator(".note-pad").fill("sync nudge " + Date.now());
await page.waitForFunction(async () => {
  const res = await fetch("/api/state");
  if (!res.ok) return false;
  const { state } = await res.json();
  return state?.tasks?.lists?.some((l) => l.items.some((i) => i.text === "E2E: buy coffee beans"));
}, null, { timeout: 20000 });
const srvRev = await page.evaluate(async () => (await fetch("/api/state")).json().then((d) => d.rev));
console.log(`    (server rev after push: ${srvRev})`);
check("local changes pushed to sync server", true);

const deviceB = await browser.newContext({ viewport: { width: 390, height: 844 } });
const phone = await deviceB.newPage();
const phoneErrors = [];
phone.on("pageerror", (err) => phoneErrors.push(String(err)));
phone.on("console", (msg) => { if (msg.type() === "error") phoneErrors.push(msg.text()); });
await phone.goto(BASE, { waitUntil: "networkidle" });
try {
  await phone.waitForFunction(() => {
    const raw = localStorage.getItem("hermesHub.v1");
    if (!raw) return false;
    const state = JSON.parse(raw);
    return state.tasks.lists.some((l) => l.items.some((i) => i.text === "E2E: buy coffee beans"));
  }, null, { timeout: 15000 });
  check("second device adopts synced state", true);
} catch {
  check("second device adopts synced state", false);
  console.log("    [debug] phone errors:", phoneErrors.slice(0, 5));
  console.log("    [debug] phone syncRev:", await phone.evaluate(() => localStorage.getItem("hermesHub.syncRev")));
  console.log("    [debug] phone has local state:", await phone.evaluate(() => !!localStorage.getItem("hermesHub.v1")));
  console.log("    [debug] phone lists:", await phone.evaluate(() => {
    const raw = localStorage.getItem("hermesHub.v1");
    if (!raw) return "none";
    return JSON.parse(raw).tasks.lists.map((l) => `${l.name}[${l.items.length}]`).join(", ");
  }));
  console.log("    [debug] server state via phone:", await phone.evaluate(async () => {
    const res = await fetch("/api/state");
    const data = await res.json();
    return `status=${res.status} rev=${data.rev} hasCoffee=${JSON.stringify(data.state || {}).includes("coffee beans")}`;
  }));
}
await phone.waitForSelector(".widget-agent");
await shot(phone, "07-mobile");
await deviceB.close();

// ---- auth lock screen (separate token-protected server) ---------------------------
if (process.env.AUTH_URL && process.env.AUTH_TOKEN) {
  const authCtx = await browser.newContext();
  const authPage = await authCtx.newPage();
  await authPage.goto(process.env.AUTH_URL, { waitUntil: "domcontentloaded" });
  await authPage.waitForSelector(".lock-backdrop", { timeout: 10000 });
  check("lock screen appears on protected server", true);
  await authPage.locator(".lock-input").fill("wrong-code");
  await authPage.locator(".lock-form .btn-primary").click();
  await authPage.waitForFunction(() =>
    document.querySelector(".lock-status")?.textContent.includes("DENIED"));
  check("wrong code denied", true);
  await authPage.locator(".lock-input").fill(process.env.AUTH_TOKEN);
  await authPage.locator(".lock-form .btn-primary").click();
  await authPage.waitForSelector(".lock-backdrop", { state: "detached", timeout: 10000 });
  await authPage.waitForSelector(".news-item", { timeout: 15000 });
  check("correct code unlocks and data loads", true);
  await shot(authPage, "08-lockscreen-unlocked");
  await authCtx.close();
} else {
  console.log("  – auth lock-screen checks skipped (set AUTH_URL + AUTH_TOKEN)");
}

// ---- console health -----------------------------------------------------------------
const realErrors = errors.filter((e) => !e.includes("favicon"));
check(`no console/page errors (${realErrors.length})`, realErrors.length === 0);
if (realErrors.length) console.log(realErrors.slice(0, 5).map((e) => "    " + e).join("\n"));

await browser.close();
console.log(failures === 0 ? "\nALL E2E CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
