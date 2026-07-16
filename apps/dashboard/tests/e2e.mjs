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
page.on("console", (msg) => {
  // server-error responses are captured (with URLs) by the response listener
  if (msg.type() === "error" && !msg.text().includes("Failed to load resource")) {
    errors.push(msg.text());
  }
});
page.on("response", (r) => {
  if (r.status() >= 500) errors.push(`${r.status()} ${r.request().method()} ${r.url()}`);
});
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
await page.evaluate(async () => {
  // clear server-side rules from previous runs
  const { rules } = await (await fetch("/api/automations")).json();
  for (const rule of rules) {
    await fetch("/api/automations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op: "delete", id: rule.id }),
    });
  }
  // clear calendar subscriptions from previous (possibly aborted) runs
  const { calendars } = await (await fetch("/api/calendars")).json();
  for (const cal of calendars) {
    await fetch("/api/calendars", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op: "remove", url: cal.url }),
    });
  }
  // ensure autonomy isn't left frozen by an aborted prior run
  await fetch("/api/killswitch", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frozen: false }),
  });
});
await page.waitForTimeout(1600); // debounced sync push of the reset
await page.reload({ waitUntil: "networkidle" });

// ---- shell -----------------------------------------------------------------
check("title", (await page.title()) === "Hermes Hub");
check("topbar brand", await page.locator(".brand-name").innerText() === "HERMES//HUB");
check("dark theme default", await page.evaluate(() => document.documentElement.dataset.theme) === "dark");

// ---- widgets render --------------------------------------------------------
for (const type of ["clock", "worldstate", "agent", "weather", "launcher", "news", "reading", "tasks", "markets", "calendar", "notes", "focus", "system"]) {
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

// structured tasks: inline !priority and @due tokens
await taskInput.fill("E2E: file taxes !high @2026-07-20");
await taskInput.press("Enter");
const prioTask = page.locator(".task-item", { hasText: "E2E: file taxes" });
check("task text strips inline tokens",
  (await prioTask.locator(".task-text").innerText()) === "E2E: file taxes");
check("high-priority task gets a priority rail",
  (await page.locator(".task-item.task-prio-high", { hasText: "file taxes" }).count()) === 1);
check("due date renders a chip", (await prioTask.locator(".task-due").count()) === 1);
check("due date persisted to state", await page.evaluate(() =>
  JSON.parse(localStorage.getItem("hermesHub.v1")).tasks.lists
    .flatMap((l) => l.items).some((t) => t.text === "E2E: file taxes" && t.due === "2026-07-20" && t.priority === "high")));

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

// open_url is confirm-tier (permission gate): an approval card appears first
await page.locator(".agent-input").fill("open GitHub");
await page.locator(".agent-form .btn-primary").click();
await page.waitForSelector(".agent-approval", { timeout: 10000 });
check("confirm-tier tool shows approval card", true);
await page.locator(".agent-approval .btn-primary").click(); // Approve
await page.waitForSelector(".viewer", { timeout: 10000 });
check("agent opens app in viewer after approval", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// deny path: add_app is confirm-tier; denying must NOT add the launcher tile
const appsBefore = await page.evaluate(() =>
  import("/js/store.js").then(({ store }) => store.state.launcher.links.length));
await page.locator(".agent-input").fill("add app DeniedApp https://denied.example.org");
await page.locator(".agent-form .btn-primary").click();
await page.waitForSelector(".agent-approval", { timeout: 10000 });
await page.locator(".agent-approval .btn:not(.btn-primary)").click(); // Deny
await page.waitForSelector(".agent-approval", { state: "detached" });
const appsAfter = await page.evaluate(() =>
  import("/js/store.js").then(({ store }) => store.state.launcher.links.length));
check("denied confirm-tier tool does not execute", appsAfter === appsBefore);
check("agent reports the declined action",
  (await page.locator(".agent-chip-err").last().innerText()).includes("declined"));

// briefing
await page.locator(".agent-quick .link-btn").first().click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => el.textContent.includes("FOCUS")), null, { timeout: 15000 });
check("briefing produced with FOCUS section", true);
check("briefing recognizes the agent-created task", await page.evaluate(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => el.textContent.includes("e2e agent made this"))));
await shot(page, "05-agent");

// ---- automations via the agent ------------------------------------------------
await page.locator(".agent-input").fill("alert me if BTC moves 0.5%");
await page.locator(".agent-form .btn-primary").click();
// create_automation is confirm-tier → approve it
await page.waitForSelector(".agent-approval", { timeout: 10000 });
await page.locator(".agent-approval .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-chip")].some((el) => el.textContent.includes("armed")),
  null, { timeout: 10000 });
check("agent creates an automation", true);

await page.locator(".agent-input").fill("list automations");
await page.locator(".agent-form .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-tool-block, .agent-chip")].some((el) => el.textContent.includes("BTC ±0.5%")),
  null, { timeout: 10000 });
check("agent lists automations", true);

// force an evaluation tick: the sample BTC 24h move (±2.41%) crosses 0.5%
const ticked = await page.evaluate(async () => {
  const res = await fetch("/api/automations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op: "tick" }),
  });
  return (await res.json()).fired;
});
check("automation fires on tick", ticked >= 1);
// nudge the poller (it also runs every 30s and on tab-focus)
await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
await page.waitForSelector(".toast", { timeout: 10000 });
check("notification surfaces as toast",
  (await page.locator(".toast").first().innerText()).includes("BTC"));

// memory: remember + recall
await page.locator(".agent-input").fill("remember: my e2e marker is zx91");
await page.locator(".agent-form .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-chip")].some((el) => el.textContent.includes("memory")),
  null, { timeout: 10000 });
await page.locator(".agent-input").fill("what do you remember?");
await page.locator(".agent-form .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => el.textContent.includes("zx91")),
  null, { timeout: 10000 });
check("agent memory persists and recalls", true);

// research tool: weather via agent
await page.locator(".agent-input").fill("what's the weather?");
await page.locator(".agent-form .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".agent-msg")].some((el) => /humidity \d+%/.test(el.textContent)),
  null, { timeout: 10000 });
check("agent answers weather from live data", true);

// ---- system status widget (telemetry) -------------------------------------------
// the agent ran several gated tools above (an approved open, a denied add_app…)
await page.locator(".widget-system .widget-controls .icon-btn[title='Refresh']").click();
await page.waitForFunction(() =>
  document.querySelectorAll(".widget-system .sys-event").length >= 1, null, { timeout: 10000 });
check("system widget shows recent tool calls", true);
check("system widget shows status rows",
  (await page.locator(".widget-system .sys-row").count()) >= 3);
// the denied add_app is counted even once it scrolls out of the recent feed
check("system widget records the denied tool",
  /[1-9]\d* denied/.test(await page.locator(".widget-system").innerText()));

// ---- kill switch (freeze all autonomy) -------------------------------------------
check("system widget shows autonomy control",
  (await page.locator(".widget-system .sys-freeze-btn").count()) === 1);
await page.locator(".widget-system .sys-freeze-btn").click(); // freeze
await page.waitForSelector(".widget-system .sys-frozen-banner", { timeout: 10000 });
check("freeze shows frozen banner", true);
const frozenTick = await page.evaluate(async () => {
  const r = await fetch("/api/automations", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify({ op: "tick" }) });
  return (await r.json()).fired;
});
check("frozen: automations do not fire on tick", frozenTick === 0);
await page.locator(".widget-system .sys-freeze-btn").click(); // resume
await page.waitForSelector(".widget-system .sys-frozen-banner", { state: "detached", timeout: 10000 });
check("resume clears the frozen banner", true);

// ---- self-evolution: agent proposals inbox (Phase 6) ----------------------------
// guarantee a pending proposal: two denials of the same tool feed reflection
await page.evaluate(async () => {
  for (let i = 0; i < 2; i++) {
    await fetch("/api/assistant/telemetry", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "add_app", tier: "confirm", ok: false, approved: false }) });
  }
});
await page.locator(".topbar-actions .menu-wrap .btn").click();
await page.locator(".menu-item", { hasText: "Agent proposals" }).click();
await page.waitForSelector(".sum-pop", { timeout: 10000 });
check("agent proposals panel opens", true);
await page.locator(".evolve-head .btn-primary").click(); // Reflect now
await page.waitForSelector(".evolve-row .evolve-actions .btn-primary", { timeout: 10000 });
check("reflection queues a pending proposal", true);
const pendingBefore = await page.locator(".evolve-row .evolve-actions").count();
await page.locator(".evolve-row .evolve-actions .btn-primary").first().click(); // Apply
await page.waitForFunction((n) =>
  document.querySelectorAll(".evolve-row .evolve-actions").length < n, pendingBefore, { timeout: 10000 });
check("applying a proposal clears it from the queue", true);
// applied proposal lands in HISTORY with a one-click rollback affordance;
// the rollback itself restores a snapshot (disruptive) so it's exercised at
// the very end of this run, below, where nothing depends on it.
check("applied proposal offers rollback in history",
  (await page.locator(".evolve-rollback").count()) >= 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// ---- streaming chat endpoint (the agent turns above already used it) ----------
const streamShape = await page.evaluate(async () => {
  const res = await fetch("/api/assistant/chat-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "what's the weather?" }], context: {} }),
  });
  const text = await res.text();
  return {
    contentType: res.headers.get("Content-Type"),
    hasDelta: text.includes("event: delta"),
    hasDone: text.includes("event: done"),
  };
});
check("chat-stream serves SSE with delta+done",
  streamShape.contentType.includes("text/event-stream") && streamShape.hasDelta && streamShape.hasDone);

// ---- reading list ---------------------------------------------------------------
await page.locator(".news-item .bookmark-btn").first().click();
await page.waitForSelector(".reading-row", { timeout: 10000 });
check("bookmark saves story to reading list", true);
const savedTitle = await page.locator(".reading-row .news-title").first().innerText();
await page.locator(".reading-row .news-item").first().click();
await page.waitForSelector(".viewer", { timeout: 10000 });
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });
await page.waitForSelector(".reading-read-chip", { timeout: 10000 });
check("opened story is marked read", true);
// the news widget redraws dimmed items on the store update; wait for it (no race)
await page.waitForFunction(() =>
  document.querySelectorAll(".widget-news .news-read").length >= 1, null, { timeout: 10000 });
check("read story dimmed in news widget", true);
await page.locator(".widget-reading .link-btn", { hasText: "Clear read" }).click();
await page.waitForFunction((t) =>
  ![...document.querySelectorAll(".reading-row .news-title")].some((el) => el.textContent === t),
  savedTitle, { timeout: 10000 });
check("clear read empties the list", true);

// ---- summarize buttons -------------------------------------------------------
check("widget summarize buttons present", (await page.locator(".widget-controls .sum-btn").count()) >= 6);
await page.locator(".news-item .sum-btn.sum-inline").first().click();
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

// ---- custom news sources -------------------------------------------------------
await page.locator(".topbar-actions .menu-wrap .btn").click();
await page.locator(".menu-item", { hasText: "News sources" }).click();
await page.waitForSelector(".sources-pop", { timeout: 10000 });
check("sources panel opens", true);
await page.locator(".sources-newtopic .input").fill("E2E Custom");
await page.locator(".sources-newtopic .btn-primary").click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".sources-topic-name")].some((el) => el.textContent === "e2e-custom"),
  null, { timeout: 10000 });
check("custom topic created", true);
const customSection = page.locator(".sources-topic", { has: page.locator(".sources-topic-name", { hasText: "e2e-custom" }) });
await customSection.locator("input[type=text]").fill("Demo Feed");
await customSection.locator("input[type=url]").fill("https://example.org/e2e-feed.xml");
await customSection.locator(".btn-primary").click();
await page.waitForSelector(".sources-row .sources-name:has-text('Demo Feed')", { timeout: 10000 });
check("feed added to custom topic", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".sources-pop", { state: "detached" });
// the news widget should now show the new tab; clicking it renders (sample fallback offline)
await page.waitForSelector(".tab:has-text('E2e Custom')", { timeout: 10000 });
await page.locator(".tab", { hasText: "E2e Custom" }).click();
await page.waitForSelector(".news-item", { timeout: 10000 });
check("custom topic tab renders stories", true);
// clean up so reruns stay deterministic
await page.evaluate(async () => {
  await fetch("/api/feeds", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op: "remove_topic", name: "e2e-custom" }) });
});
await page.locator(".tab", { hasText: "Top" }).click();

// ---- weather extras (AQI + sun, from sample data offline) -----------------------
check("weather sun line shown", /☀ \d\d:\d\d · ☾ \d\d:\d\d/.test(
  await page.locator(".weather-extras").innerText()));
check("weather AQI chip shown", /AQI \d+/.test(await page.locator(".aqi-chip").innerText()));

// ---- calendar feeds (ICS subscriptions) ------------------------------------------
await page.locator(".topbar-actions .menu-wrap .btn").click();
await page.locator(".menu-item", { hasText: "Calendar feeds" }).click();
await page.waitForSelector(".sources-pop", { timeout: 10000 });
check("calendar feeds panel opens", true);
await page.locator("input[aria-label='Calendar name']").fill("Demo");
await page.locator("input[aria-label='Calendar URL']").fill(`${BASE}/demo.ics`);
await page.locator(".sources-add .btn-primary").click();
await page.waitForSelector(".sources-row .sources-name:has-text('Demo')", { timeout: 10000 });
check("calendar subscribed", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".sources-pop", { state: "detached" });
// demo.ics has a daily 07:00 event → today (selected by default) shows it read-only
await page.waitForSelector(".cal-event-ext", { timeout: 10000 });
check("external event listed for today", (await page.locator(".cal-event-ext").allInnerTexts()).join(" ").includes("Demo: morning run"));
// .cal-event-cal renders uppercase (text-transform), innerText reflects that
check("external event shows calendar name", (await page.locator(".cal-event-cal").first().innerText()).toUpperCase() === "DEMO");
check("upcoming merges subscribed calendar", (await page.locator(".cal-upcoming").innerText()).includes("(Demo)"));
await shot(page, "09-calendar-feeds");
// unsubscribe so the rest of the run (and reruns) see a clean calendar
await page.evaluate(async (base) => {
  await fetch("/api/calendars", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op: "remove", url: `${base}/demo.ics` }) });
  window.dispatchEvent(new CustomEvent("hub:calendars-changed"));
}, BASE);
await page.waitForSelector(".cal-event-ext", { state: "detached", timeout: 10000 });
check("unsubscribe clears external events", true);

// ---- crypto detail drawer (chart + indicators) ----------------------------------
await page.waitForSelector(".market-row");
await page.locator(".market-row").first().click();
await page.waitForSelector(".detail-pop .coin-price", { timeout: 8000 });
check("coin detail opens with price", /\$[\d,]/.test(await page.locator(".coin-price").innerText()));
check("coin detail renders a candle chart", (await page.locator(".coin-chart-wrap rect").count()) > 10);
check("coin detail shows technical signals", (await page.locator(".coin-signal").count()) >= 3);
check("coin detail shows stat grid", (await page.locator(".coin-stat").count()) >= 6);
// range switch re-renders the chart without closing the drawer
await page.locator(".detail-pop .tab", { hasText: "7D" }).click();
await page.waitForFunction(() =>
  document.querySelector('.detail-pop .tab[aria-selected="true"]')?.textContent === "7D",
  null, { timeout: 5000 });
check("coin detail range switch works", (await page.locator(".coin-chart-wrap rect").count()) > 10);
await page.keyboard.press("Escape");
await page.waitForSelector(".detail-pop", { state: "detached" });
check("coin detail closes", true);

// ---- markets watchlist editor ---------------------------------------------------
await page.waitForSelector(".market-row");
const marketCountBefore = await page.locator(".market-row").count();
page.once("dialog", (dialog) => dialog.accept("solana"));
// remove SOL first (edit mode), then re-add it via the prompt
await page.locator("#edit-toggle").click();
await page.waitForSelector(".market-row .icon-btn[title='Remove from watchlist']");
await page.locator(".market-row", { hasText: "SOL" }).locator(".icon-btn[title='Remove from watchlist']").click();
await page.waitForFunction((n) => document.querySelectorAll(".market-row").length === n - 1,
  marketCountBefore, { timeout: 10000 });
check("watchlist remove works", true);
await page.locator(".market-note-row .link-btn").click(); // prompt answered above
await page.waitForFunction((n) => document.querySelectorAll(".market-row").length === n,
  marketCountBefore, { timeout: 10000 });
check("watchlist add works", true);
await page.locator("#edit-toggle").click();

// ---- voice controls (presence + graceful degradation) ----------------------------
check("voice replies toggle present when supported", await page.evaluate(() =>
  !("speechSynthesis" in window)
  || [...document.querySelectorAll(".agent-quick .link-btn")].some((el) => el.textContent.includes("Voice"))));
check("mic hidden when SpeechRecognition unsupported", await page.evaluate(() =>
  (("SpeechRecognition" in window) || ("webkitSpeechRecognition" in window))
    ? document.querySelector(".agent-mic") !== null
    : document.querySelector(".agent-mic") === null));

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
// palette also searches personal data (the "E2E: buy coffee beans" task exists by now)
await page.locator(".palette-input").fill("coffee");
await page.waitForFunction(() =>
  [...document.querySelectorAll(".palette-item")].some((el) => /coffee/i.test(el.textContent) && /task/i.test(el.textContent)),
  null, { timeout: 5000 });
check("palette searches personal data (tasks)", true);
await page.locator(".palette-item", { hasText: /coffee/i }).first().click();
await page.waitForSelector(".widget-tasks.widget-flash", { timeout: 3000 });
check("palette jump flashes the target widget", true);

// palette can run typed text as an agent command (reuses the agent loop)
await page.keyboard.press("Control+k");
await page.waitForSelector(".palette", { timeout: 5000 });
await page.locator(".palette-input").fill("add task E2E palette wins");
await page.waitForFunction(() =>
  [...document.querySelectorAll(".palette-item")].some((el) => /as a command/i.test(el.textContent)),
  null, { timeout: 5000 });
check("palette offers run-as-command", true);
await page.locator(".palette-item", { hasText: /as a command/i }).click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".task-text")].some((el) => el.textContent.includes("E2E palette wins")),
  null, { timeout: 8000 });
check("palette runs command through the agent", true);

// ---- focus timer ----------------------------------------------------------------
await page.waitForSelector(".widget-focus .focus-clock", { timeout: 10000 });
check("focus timer shows 25:00 idle",
  (await page.locator(".widget-focus .focus-clock").innerText()).trim() === "25:00");
await page.locator(".widget-focus .focus-controls .btn-primary").click(); // Start
await page.waitForFunction(() =>
  /RUNNING/.test(document.querySelector(".widget-focus .focus-mode")?.textContent || ""),
  null, { timeout: 3000 });
check("focus timer starts running", true);
await page.locator(".widget-focus .focus-presets .note-tab", { hasText: "Break 5" }).click();
check("focus preset switches to 05:00 break",
  (await page.locator(".widget-focus .focus-clock").innerText()).trim() === "05:00");
check("focus break mode styled", (await page.locator(".widget-focus .focus-clock.focus-break").count()) === 1);

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

// ---- news search (client-side filter, no refetch) ---------------------------
// Run late so the transient toasts these steps raise can't race earlier checks.
await page.locator(".tab", { hasText: "Top" }).click();
await page.waitForSelector(".news-item");
const newsBefore = await page.locator(".news-item").count();
const firstHeadline = (await page.locator(".news-title").first().innerText()).split(" ")[0];
await page.locator(".news-search").fill(firstHeadline);
await page.waitForFunction(
  (n) => document.querySelectorAll(".news-item").length <= n,
  newsBefore, { timeout: 4000 });
check("news search filters results", (await page.locator(".news-item").count()) <= newsBefore);
await page.locator(".news-search").fill("zzzznotarealheadlinexyzzy");
await page.waitForFunction(() => document.querySelectorAll(".news-item").length === 0, null, { timeout: 4000 });
check("news search shows empty state on no match",
  (await page.locator(".news-list .muted").count()) >= 1);
await page.locator(".news-search").fill("");
await page.waitForSelector(".news-item");
check("clearing news search restores items", (await page.locator(".news-item").count()) > 2);

// ---- accent presets ----------------------------------------------------------
await page.locator(".menu-wrap > .btn").click();
await page.locator(".accent-swatch[aria-label='Accent amber']").click();
check("accent preset applies", await page.evaluate(() =>
  getComputedStyle(document.documentElement).getPropertyValue("--accent").trim().toLowerCase() === "#f2b13c"));
check("accent persists to state", await page.evaluate(() =>
  JSON.parse(localStorage.getItem("hermesHub.v1"))?.accent === "amber"));
await page.locator(".menu-wrap > .btn").click();
await page.locator(".accent-swatch[aria-label='Accent cyan']").click();
check("accent reset clears override", await page.evaluate(() =>
  document.documentElement.style.getPropertyValue("--accent") === ""));

// ---- server backup download + import roundtrip ------------------------------
const backupOk = await page.evaluate(async () => {
  const post = (url, body) => fetch(url, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
  const mk = await post("/api/backup", {});
  const snap = await (await fetch("/api/backup/get?name=" + mk.name)).json();
  if (snap.kind !== "hermes-hub-backup") return false;
  const imp = await post("/api/backup/import", { snapshot: snap });
  return /^hub-/.test(imp.name || "");
});
check("server backup download+import roundtrip", backupOk === true);
await page.locator(".menu-wrap > .btn").click();
check("server backup menu items present",
  (await page.locator(".menu-item", { hasText: "Download server backup" }).count()) === 1 &&
  (await page.locator(".menu-item", { hasText: "Restore server backup" }).count()) === 1);
await page.locator(".menu-wrap > .btn").click(); // toggle the menu closed again

// ---- model routing overrides (Phase 1 UI) -----------------------------------
await page.locator(".menu-wrap > .btn").click();
await page.locator(".menu-item", { hasText: "Model routing" }).click();
await page.waitForSelector(".routing-row", { timeout: 10000 });
check("routing panel shows three tiers", (await page.locator(".routing-row").count()) === 3);
const coreInput = page.locator(".routing-row", { hasText: "CORE" }).locator(".routing-input");
await coreInput.fill("claude-sonnet-test");
await page.locator(".routing-actions .btn-primary").click();
await page.waitForFunction(() => document.querySelector(".routing-input") &&
  [...document.querySelectorAll(".routing-row")].some((r) => r.textContent.includes("claude-sonnet-test")),
  null, { timeout: 10000 });
check("routing override saved and active", true);
const routingCore = await page.evaluate(async () =>
  (await (await fetch("/api/assistant/routing")).json()).overrides.core);
check("routing override persisted server-side", routingCore === "claude-sonnet-test");
await page.locator(".routing-actions .btn", { hasText: "Reset" }).click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".routing-input")].every((i) => !i.value),
  null, { timeout: 10000 });
check("routing reset clears overrides", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// ---- evolution rollback (Phase 6 audit) -------------------------------------
// Runs last: rollback restores a snapshot, which advances the sync rev and
// re-renders the board, so nothing downstream may depend on it.
await page.locator(".menu-wrap > .btn").click();
await page.locator(".menu-item", { hasText: "Agent proposals" }).click();
await page.waitForSelector(".evolve-rollback", { timeout: 10000 });
await page.locator(".evolve-rollback").first().click();
await page.waitForFunction(() =>
  [...document.querySelectorAll(".evolve-status")].some((el) => el.textContent.includes("ROLLED BACK")),
  null, { timeout: 10000 });
check("rollback marks the proposal rolled back", true);
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// ---- console health -----------------------------------------------------------------
const realErrors = errors.filter((e) => !e.includes("favicon"));
check(`no console/page errors (${realErrors.length})`, realErrors.length === 0);
if (realErrors.length) console.log(realErrors.slice(0, 5).map((e) => "    " + e).join("\n"));

await browser.close();
console.log(failures === 0 ? "\nALL E2E CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
