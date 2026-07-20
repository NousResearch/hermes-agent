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

// Dashboard is split into pages; switch to the page holding a widget before
// interacting with it. gotoPage clicks the page tab and settles.
const WIDGET_PAGES = {
  Main: ["glance", "clock", "worldstate", "agent", "weather", "launcher", "tasks", "calendar", "notes", "focus", "system"],
  Markets: ["markets", "stocks", "commodities"],
  Feeds: ["news", "reading", "socials", "gaming", "podcasts"],
  Sports: ["scores", "racing"],
  Intel: ["worldclock", "quakes", "fx", "convert", "air", "marine", "space", "alerts", "flights"],
  Health: ["medbot", "pubmed", "trials", "drug", "calc", "meded"],
  "AI Lab": ["codelab", "ailearn", "snippets", "repos", "papers", "ainews", "aidaily"],
};
const pageOf = (type) => Object.keys(WIDGET_PAGES).find((p) => WIDGET_PAGES[p].includes(type)) || "Main";
const gotoPage = async (name) => {
  await page.locator(".pagetab", { hasText: new RegExp(`^${name}$`) }).click();
  await page.waitForTimeout(60);
};
const gotoWidget = async (type) => { await gotoPage(pageOf(type)); await page.waitForSelector(`.widget-${type}`, { timeout: 10000 }); };

// ---- widgets render (per page) ---------------------------------------------
check("page tabs render", (await page.locator(".pagetab").count()) >= 3);
for (const [pageName, types] of Object.entries(WIDGET_PAGES)) {
  await gotoPage(pageName);
  for (const type of types) {
    await page.waitForSelector(`.widget-${type}`, { timeout: 10000 });
    check(`widget ${type} present`, true);
  }
}
await gotoPage("Main");
await page.waitForFunction(() => document.querySelectorAll(".widget-glance .glance-cell").length >= 5, null, { timeout: 10000 });
check("at-a-glance hero renders cells", (await page.locator(".widget-glance .glance-cell").count()) >= 5);
check("glance weather cell populated", /\d+°/.test(await page.locator(".widget-glance").innerText()));
check("clock shows time", /\d{1,2}:\d{2}/.test(await page.locator(".clock-time").innerText()));
check("weather temp shown", /-?\d+°/.test(await page.locator(".weather-temp").innerText()));
await page.waitForFunction(() => document.querySelectorAll(".ws-row").length >= 5, null, { timeout: 10000 });
check("worldstate domains", (await page.locator(".ws-row").count()) >= 5);
check("worldstate levels are labeled", (await page.locator(".ws-row .level-chip").first().innerText()).length > 2);
check("sample badges when offline", (await page.locator(".widget-badge:not([hidden])").count()) >= 2);
await gotoWidget("news");
await page.waitForSelector(".news-item", { timeout: 10000 });
check("news items rendered", (await page.locator(".news-item").count()) > 3);
await gotoWidget("markets");
check("market rows", (await page.locator(".market-row").count()) >= 3);
await gotoPage("Main");
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
await gotoWidget("news");
await page.locator(".news-item").first().click();
await page.waitForSelector(".viewer", { timeout: 5000 });
check("news opens in-app viewer", true);
check("viewer has reader/embed tabs", (await page.locator(".viewer-tab").count()) === 2);
await page.locator(".viewer-tab", { hasText: "EMBED" }).click();
check("embed tab shows iframe", (await page.locator(".viewer-frame").count()) === 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// ---- launcher opens in-app --------------------------------------------------
await gotoWidget("launcher");
await page.locator(".app-tile").first().click();
await page.waitForSelector(".viewer", { timeout: 5000 });
check("app tile opens in-app viewer (embed)", (await page.locator(".viewer-frame").count()) === 1);
await page.keyboard.press("Escape");
await page.waitForSelector(".viewer", { state: "detached" });

// ---- news topic tabs ---------------------------------------------------------
await gotoWidget("news");
await page.locator(".widget-news .tab", { hasText: "Science" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-news .tab[aria-selected="true"]')?.textContent === "Science");
await page.waitForSelector(".news-item");
check("topic switch renders items", (await page.locator(".news-item").count()) > 2);

// mute a source → its items disappear, an unmute chip appears
const firstSource = await page.locator(".widget-news .news-item .news-source").first().innerText();
const beforeMute = await page.locator(".widget-news .news-item").count();
await page.locator(".widget-news .news-item").first().locator(".news-mute").click();
await page.waitForSelector(".widget-news .news-muted-bar", { timeout: 5000 });
check("muting a source hides its items",
  (await page.locator(".widget-news .news-item").count()) < beforeMute);
check("muted source shown in unmute bar",
  (await page.locator(".widget-news .news-muted-chip").innerText()).toUpperCase().includes(firstSource.toUpperCase()));
await page.locator(".widget-news .news-muted-chip").first().click();
await page.waitForSelector(".widget-news .news-muted-bar", { state: "detached", timeout: 5000 });
check("unmuting restores items",
  (await page.locator(".widget-news .news-item").count()) >= beforeMute);

// gaming is a first-class news topic
await page.locator(".widget-news .tab", { hasText: "Gaming" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-news .tab[aria-selected="true"]')?.textContent === "Gaming",
  null, { timeout: 5000 });
await page.waitForSelector(".news-item");
check("gaming news topic renders", (await page.locator(".news-item").count()) >= 2);

// ---- news topic detail window (⤢) + cross-topic search ----------------------
await page.locator(".widget-news .widget-expand").click();
await page.waitForSelector(".news-detail-grid .news-detail-item", { timeout: 5000 });
check("news detail groups items by source", (await page.locator(".news-detail-col").count()) >= 1);
const detailCount = await page.locator(".news-detail-item").count();
check("news detail lists items", detailCount >= 2);
await page.locator(".news-all-toggle input").check();
await page.waitForFunction(
  () => document.querySelector(".detail-scope")?.textContent === "ALL TOPICS", null, { timeout: 5000 });
await page.waitForSelector(".news-topic-tag", { timeout: 5000 });
const tags = await page.locator(".news-topic-tag").allInnerTexts();
check("cross-topic search spans ≥2 topics", new Set(tags).size >= 2);
await page.keyboard.press("Escape");
await page.waitForSelector(".detail-backdrop", { state: "detached" });

// ---- tasks -------------------------------------------------------------------
await gotoWidget("tasks");
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
check("proposal shows a provenance badge",
  (await page.locator(".evolve-row .evolve-source").first().innerText()).replace(/\s/g, "").length >= 3);
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
await gotoWidget("news");
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
await gotoWidget("news");
check("widget summarize buttons present", (await page.locator(".widget-controls .sum-btn").count()) >= 3);
await page.locator(".news-item .sum-btn.sum-inline").first().click();
await page.waitForSelector(".sum-pop", { timeout: 10000 });
await page.waitForFunction(() => document.querySelector(".sum-body p"));
check("news item summary shows text", (await page.locator(".sum-body").innerText()).length > 20);
check("summary labeled with engine", ["LOCAL", "CLAUDE"].includes(await page.locator(".sum-mode").innerText()));
await shot(page, "06-summary");
await page.keyboard.press("Escape");
await page.waitForSelector(".sum-pop", { state: "detached" });

// widget-level summarize (worldstate)
await gotoWidget("worldstate");
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
await gotoWidget("news");
await page.waitForSelector(".widget-news .tab:has-text('E2e Custom')", { timeout: 10000 });
await page.locator(".widget-news .tab", { hasText: "E2e Custom" }).click();
await page.waitForSelector(".news-item", { timeout: 10000 });
check("custom topic tab renders stories", true);
// follow-a-search creates a Google News topic
const followOk = await page.evaluate(async () => {
  const post = (body) => fetch("/api/feeds", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
  const snap = await post({ op: "add_search", name: "E2E Search", query: "quantum computing" });
  const key = Object.keys(snap.sources).find((k) => k.includes("e2e-search"));
  const ok = key && snap.sources[key][0].url.includes("news.google.com/rss/search");
  await post({ op: "remove_topic", name: "e2e-search" }); // cleanup
  return ok;
});
check("follow-a-search creates a Google News topic", followOk === true);
// clean up so reruns stay deterministic
await page.evaluate(async () => {
  await fetch("/api/feeds", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op: "remove_topic", name: "e2e-custom" }) });
});
await page.locator(".widget-news .tab", { hasText: "Top" }).click();

// ---- weather extras (AQI + sun, from sample data offline) -----------------------
await gotoWidget("weather");
check("weather sun line shown", /☀ \d\d:\d\d · ☾ \d\d:\d\d/.test(
  await page.locator(".weather-extras").innerText()));
check("weather AQI chip shown", /AQI \d+/.test(await page.locator(".aqi-chip").innerText()));

// ---- calendar feeds (ICS subscriptions) ------------------------------------------
await gotoWidget("calendar");
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

// ---- stocks / indices / FX -------------------------------------------------------
await gotoWidget("stocks");
await page.waitForSelector(".widget-stocks .market-row");
check("stocks rows render", (await page.locator(".widget-stocks .market-row").count()) >= 3);
await page.locator(".widget-stocks .market-row").first().click();
await page.waitForSelector(".detail-pop .coin-chart-wrap rect", { timeout: 8000 });
check("stock detail chart renders", (await page.locator(".detail-pop .coin-chart-wrap rect").count()) > 10);
check("stock detail shows signals", (await page.locator(".detail-pop .coin-signal").count()) >= 3);
await page.keyboard.press("Escape");
await page.waitForSelector(".detail-pop", { state: "detached" });

// ---- gaming (free games + deals) -------------------------------------------------
await gotoWidget("gaming");
await page.waitForSelector(".widget-gaming .game-free, .widget-gaming .game-deal");
check("gaming free games render", (await page.locator(".widget-gaming .game-free").count()) >= 1);
check("gaming steam deals render", (await page.locator(".widget-gaming .game-deal").count()) >= 1);

// ---- socials hub -----------------------------------------------------------------
await gotoWidget("socials");
await page.waitForSelector(".widget-socials .social-item");
check("socials feed renders items", (await page.locator(".widget-socials .social-item").count()) >= 1);
await page.locator(".widget-socials .tab", { hasText: "Reddit" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-socials .tab[aria-selected="true"]')?.textContent === "Reddit",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-socials .social-item");
check("socials network switch works", (await page.locator(".widget-socials .social-badge").first().innerText()) === "RE");

// ---- sports scores ---------------------------------------------------------------
await gotoWidget("scores");
await page.waitForSelector(".widget-scores .score-game");
check("scores board renders games", (await page.locator(".widget-scores .score-game").count()) >= 1);
check("scores show a status chip", (await page.locator(".widget-scores .score-chip").count()) >= 1);
await page.locator(".widget-scores .tab", { hasText: "NFL" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-scores .tab[aria-selected="true"]')?.textContent === "NFL",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-scores .score-game");
check("scores league switch works", (await page.locator(".widget-scores .score-game").count()) >= 1);
// rugby league (SA franchises)
await page.locator(".widget-scores .tab", { hasText: "URC" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-scores .tab[aria-selected="true"]')?.textContent === "URC",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-scores .score-game");
check("rugby URC board renders", (await page.locator(".widget-scores .score-game").count()) >= 1);
// tennis (individual-sport board — no home/away flags)
await page.locator(".widget-scores .tab", { hasText: "Tennis ATP" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-scores .tab[aria-selected="true"]')?.textContent === "Tennis ATP",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-scores .score-game");
check("tennis ATP board renders", (await page.locator(".widget-scores .score-game").count()) >= 1);
// back to NBA for the standings detail check (rugby standings vary)
await page.locator(".widget-scores .tab", { hasText: "NBA" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-scores .tab[aria-selected="true"]')?.textContent === "NBA",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-scores .score-game");
// standings detail window
await page.locator(".widget-scores .widget-expand").click();
await page.waitForSelector(".detail-pop .stand-row", { timeout: 8000 });
check("standings table renders", (await page.locator(".detail-pop .stand-row").count()) >= 3);
await page.keyboard.press("Escape");
await page.waitForSelector(".detail-pop", { state: "detached" });

// ---- follow teams → My Teams tab + add fixture to calendar ----------------------
await page.locator(".widget-scores .score-follow").first().click();
await page.waitForSelector(".widget-scores .score-follow.score-followed", { timeout: 5000 });
check("following a team fills the star", true);
await page.locator(".widget-scores .score-teams-tab").click();
await page.waitForSelector(".widget-scores .myteam", { timeout: 5000 });
check("My Teams tab lists a followed team", (await page.locator(".widget-scores .myteam").count()) >= 1);
await page.waitForSelector(".widget-scores .myteam-cal", { timeout: 5000 });
await page.locator(".widget-scores .myteam-cal").first().click();
await page.waitForFunction(() =>
  /Added/.test(document.querySelector(".widget-scores .myteam-cal")?.textContent || ""),
  null, { timeout: 5000 });
check("add-to-calendar marks the fixture added", true);

// ---- motorsport (ESPN racing) ----------------------------------------------------
await gotoWidget("racing");
await page.waitForSelector(".widget-racing .race-card", { timeout: 5000 });
check("motorsport lists races", (await page.locator(".widget-racing .race-card").count()) >= 1);
check("motorsport shows a podium", (await page.locator(".widget-racing .race-podium").count()) >= 1);
await page.locator(".widget-racing .tab", { hasText: "MotoGP" }).click();
await page.waitForFunction(() =>
  document.querySelector('.widget-racing .tab[aria-selected="true"]')?.textContent === "MotoGP",
  null, { timeout: 5000 });
await page.waitForSelector(".widget-racing .race-card");
check("motorsport series switch works", (await page.locator(".widget-racing .race-card").count()) >= 1);

// ---- air quality + pollen --------------------------------------------------------
await gotoWidget("air");
await page.waitForSelector(".widget-air .aq-aqi", { timeout: 5000 });
check("air quality gauge shows an AQI", /\d/.test(await page.locator(".widget-air .aq-aqi").innerText()));
check("air quality band labeled", (await page.locator(".widget-air .aq-band").innerText()).length > 2);
check("air quality lists pollutants", (await page.locator(".widget-air .aq-cell").count()) >= 3);
check("air quality shows pollen rows", (await page.locator(".widget-air .aq-pollen-row").count()) >= 1);

// ---- marine & surf (Open-Meteo Marine) -------------------------------------------
await gotoWidget("marine");
await page.waitForSelector(".widget-marine .mar-wave", { timeout: 5000 });
check("marine shows a wave height", /\d/.test(await page.locator(".widget-marine .mar-wave").innerText()));
check("marine labels a sea state", (await page.locator(".widget-marine .mar-state").innerText()).length > 2);
check("marine lists conditions", (await page.locator(".widget-marine .mar-cell").count()) >= 5);

// ---- space weather (NOAA SWPC) ---------------------------------------------------
await gotoWidget("space");
await page.waitForSelector(".widget-space .sw-kp", { timeout: 5000 });
check("space weather shows a Kp reading", /Kp\s+[\d.]/.test(await page.locator(".widget-space .sw-kp").innerText()));
check("space weather band labeled", (await page.locator(".widget-space .sw-band").innerText()).length > 2);
check("space weather renders Kp history bars", (await page.locator(".widget-space .sw-bar").count()) >= 3);

// ---- weather alerts (NWS) --------------------------------------------------------
await gotoWidget("alerts");
await page.waitForSelector(".widget-alerts .wa-card, .widget-alerts .wa-clear", { timeout: 5000 });
check("weather alerts render cards", (await page.locator(".widget-alerts .wa-card").count()) >= 1);
check("alert shows a severity label", (await page.locator(".widget-alerts .wa-sev").first().innerText()).length > 2);

// ---- flights overhead (OpenSky) --------------------------------------------------
await gotoWidget("flights");
await page.waitForSelector(".widget-flights .fl-row", { timeout: 5000 });
check("flights list renders aircraft", (await page.locator(".widget-flights .fl-row").count()) >= 3);
check("flight shows a callsign", (await page.locator(".widget-flights .fl-call").first().innerText()).length > 1);
check("flight shows an altitude", /ft/.test(await page.locator(".widget-flights .fl-alt").first().innerText()));

// ---- health & medicine (PubMed, trials, SA MedBot) -------------------------------
await gotoWidget("pubmed");
await page.waitForSelector(".widget-pubmed .pubmed-item", { timeout: 10000 });
check("pubmed lists recent articles", (await page.locator(".widget-pubmed .pubmed-item").count()) >= 2);
await gotoWidget("trials");
check("clinical trials list renders", (await page.locator(".widget-trials .trial-item").count()) >= 2);
check("trial shows a status chip", (await page.locator(".widget-trials .trial-status").count()) >= 1);
await gotoWidget("drug");
await page.waitForSelector(".widget-drug .drug-name", { timeout: 5000 });
check("drug reference shows a medication name", (await page.locator(".widget-drug .drug-name").innerText()).length > 2);
check("drug reference lists label sections", (await page.locator(".widget-drug .drug-section").count()) >= 2);
// collapsible: the first section's panel toggles
const panelHiddenBefore = await page.locator(".widget-drug .drug-panel").first().isHidden();
await page.locator(".widget-drug .drug-section-btn").first().click();
await page.waitForTimeout(150);
check("drug section toggles on click",
  (await page.locator(".widget-drug .drug-panel").first().isHidden()) !== panelHiddenBefore);
check("drug widget links SA guidelines", (await page.locator(".widget-drug .drug-sa").count()) >= 1);
// bridge: "Ask SA MedBot for dosing" hands the question to the MedBot on the same page
const medMsgsBefore = await page.locator(".widget-medbot .med-msg").count();
await page.locator(".widget-drug .drug-ask").click();
await page.waitForFunction((n) =>
  document.querySelectorAll(".widget-medbot .med-msg").length > n, medMsgsBefore, { timeout: 10000 });
check("drug → MedBot bridge asks a SA dosing question",
  (await page.locator(".widget-medbot .med-msg.med-user").first().innerText()).toLowerCase().includes("south african"));
// clinical calculators
await gotoWidget("calc");
await page.locator(".widget-calc .calc-inputs input").first().fill("60");
await page.locator(".widget-calc .calc-field select").first().selectOption("male");
await page.locator(".widget-calc .calc-inputs input").nth(1).fill("90");
await page.waitForFunction(() =>
  /\d/.test(document.querySelector(".widget-calc .calc-val")?.textContent || ""), null, { timeout: 5000 });
check("clinical calc computes eGFR", /\d/.test(await page.locator(".widget-calc .calc-val").innerText()));
check("clinical calc shows interpretation", (await page.locator(".widget-calc .calc-interp").innerText()).toLowerCase().includes("ckd"));
// GCS uses labelled selects (E/V/M) → 15
await page.locator(".widget-calc .calc-picker").selectOption("gcs");
await page.waitForSelector(".widget-calc .calc-field select", { timeout: 5000 });
const gcsSel = page.locator(".widget-calc .calc-field select");
await gcsSel.nth(0).selectOption("4");
await gcsSel.nth(1).selectOption("5");
await gcsSel.nth(2).selectOption("6");
await page.waitForFunction(() =>
  document.querySelector(".widget-calc .calc-val")?.textContent === "15", null, { timeout: 5000 });
check("clinical calc computes GCS from labelled selects", true);
await page.locator(".widget-calc .calc-picker").selectOption("reference");
await page.waitForSelector(".widget-calc .calc-ref-table tr", { timeout: 5000 });
check("clinical calc lists SA reference ranges", (await page.locator(".widget-calc .calc-ref-table tr").count()) >= 10);
// Naegele EDD (date input support)
await page.locator(".widget-calc .calc-picker").selectOption("naegele");
await page.locator(".widget-calc .calc-field input[type=date]").fill("2026-01-01");
await page.waitForFunction(() =>
  /2026-10/.test(document.querySelector(".widget-calc .calc-val")?.textContent || ""),
  null, { timeout: 5000 });
check("clinical calc computes Naegele EDD", /2026-10-08/.test(await page.locator(".widget-calc .calc-val").innerText()));
// Alvarado appendicitis score (checkbox scoring)
await page.locator(".widget-calc .calc-picker").selectOption("alvarado");
await page.waitForSelector(".widget-calc .calc-check", { timeout: 5000 });
for (const cb of await page.locator(".widget-calc .calc-check input").all()) await cb.check();
await page.waitForFunction(() =>
  document.querySelector(".widget-calc .calc-val")?.textContent === "10", null, { timeout: 5000 });
check("clinical calc scores Alvarado", (await page.locator(".widget-calc .calc-val").innerText()) === "10");
// med education / OSCE
await gotoWidget("meded");
check("med ed lists OSCE stations", (await page.locator(".widget-meded .meded-station").count()) >= 10);
const medMsgsB = await page.locator(".widget-medbot .med-msg").count();
await page.locator(".widget-meded .meded-station .btn-primary").first().click();
await page.waitForFunction((n) =>
  document.querySelectorAll(".widget-medbot .med-msg").length > n, medMsgsB, { timeout: 10000 });
check("OSCE 'Practice' launches a MedBot station",
  (await page.locator(".widget-medbot .med-msg.med-user").last().innerText()).toLowerCase().includes("osce"));
await page.locator(".widget-meded .meded-modes .tab", { hasText: "Study" }).click();
await page.waitForSelector(".widget-meded .meded-card", { timeout: 5000 });
check("med ed study cards render", (await page.locator(".widget-meded .meded-card").count()) >= 4);

// ---- AI Lab: code runner + resources + prompt library ---------------------------
await gotoWidget("codelab");
await page.waitForSelector(".widget-codelab .cl-editor", { timeout: 5000 });
await page.evaluate(() => {
  const t = document.querySelector(".widget-codelab .cl-editor");
  t.value = "function solution(nums,target){const m=new Map();for(let i=0;i<nums.length;i++){if(m.has(target-nums[i]))return [m.get(target-nums[i]),i];m.set(nums[i],i);}}";
  t.dispatchEvent(new Event("input", { bubbles: true }));
});
await page.locator(".widget-codelab .cl-actions .btn-primary").click();
await page.waitForSelector(".widget-codelab .cl-summary.cl-pass", { timeout: 5000 });
check("code lab runs JS in a worker and passes tests",
  (await page.locator(".widget-codelab .cl-summary").innerText()).includes("passed"));
await page.locator(".widget-codelab .cl-actions .btn", { hasText: "Hint" }).click();
check("code lab reveals a hint", (await page.locator(".widget-codelab .cl-hint").count()) >= 1);
check("AI hub shows resources + daily pick",
  (await page.locator(".widget-ailearn .ail-item").count()) >= 4 && (await page.locator(".widget-ailearn .ail-daily").count()) === 1);
check("prompt library seeded", (await page.locator(".widget-snippets .snip-item").count()) >= 3);
await page.locator(".widget-snippets .snip-search").fill("debounce");
await page.waitForTimeout(150);
check("prompt library search filters", (await page.locator(".widget-snippets .snip-item").count()) === 1);
// Repo Radar + arXiv Papers (live data, sample fallback offline)
await gotoWidget("repos");
await page.waitForSelector(".widget-repos .repo-item", { timeout: 5000 });
check("repo radar lists repositories", (await page.locator(".widget-repos .repo-item").count()) >= 3);
check("repo shows a star count", /★/.test(await page.locator(".widget-repos .repo-stars").first().innerText()));
await gotoWidget("commodities");
await page.waitForSelector(".widget-commodities .commod-row", { timeout: 5000 });
check("commodities lists priced rows", (await page.locator(".widget-commodities .commod-row").count()) >= 5);
check("commodities groups metals & rates",
  (await page.locator(".widget-commodities .commod-group-label").count()) >= 3);
await gotoWidget("papers");
await page.waitForSelector(".widget-papers .paper-item", { timeout: 5000 });
check("arxiv papers list renders", (await page.locator(".widget-papers .paper-item").count()) >= 2);
await page.locator(".widget-papers .paper-head").first().click();
await page.waitForTimeout(150);
check("paper abstract expands", !(await page.locator(".widget-papers .paper-abstract").first().isHidden()));
await gotoWidget("ainews");
await page.waitForSelector(".widget-ainews .news-item", { timeout: 5000 });
check("AI radar lists stories", (await page.locator(".widget-ainews .news-item").count()) >= 2);
await gotoWidget("aidaily");
await page.waitForSelector(".widget-aidaily .aid-card", { timeout: 5000 });
check("AI daily brief shows category picks", (await page.locator(".widget-aidaily .aid-card").count()) >= 4);
await gotoWidget("medbot");
check("medbot shows the SA decision-support intro", /South African/i.test(await page.locator(".widget-medbot").innerText()));
await page.locator(".widget-medbot .med-input").fill("First-line HIV-TB co-infection management?");
await page.locator(".widget-medbot .med-form .btn-primary").click();
await page.waitForFunction(() =>
  document.querySelectorAll(".widget-medbot .med-msg").length >= 2, null, { timeout: 10000 });
check("medbot answers a clinical question", (await page.locator(".widget-medbot .med-msg").count()) >= 2);
// medicine is a news topic
await gotoWidget("news");
await page.locator(".widget-news .tab", { hasText: "Medicine" }).click();
await page.waitForSelector(".news-item");
check("medicine news topic renders", (await page.locator(".news-item").count()) >= 2);

// ---- intel widgets (world clock, seismic, currency) ------------------------------
await gotoWidget("worldclock");
check("world clock shows zones", (await page.locator(".widget-worldclock .wc-row").count()) >= 3);
check("world clock shows a time", /\d\d:\d\d/.test(await page.locator(".widget-worldclock .wc-time").first().innerText()));
await gotoWidget("quakes");
check("seismic monitor lists quakes", (await page.locator(".widget-quakes .quake-row").count()) >= 3);
check("quake magnitude shown", /\d\.\d/.test(await page.locator(".widget-quakes .quake-mag").first().innerText()));
await gotoWidget("fx");
check("currency rows render", (await page.locator(".widget-fx .fx-row").count()) >= 3);
await page.locator(".widget-fx .fx-amount").fill("200");
await page.waitForTimeout(100);
check("currency recomputes on amount change",
  parseFloat((await page.locator(".widget-fx .fx-val").first().innerText()).replace(/,/g, "")) > 0);

await gotoWidget("convert");
await page.waitForSelector(".widget-convert .cv-out");
const cvFirst = parseFloat((await page.locator(".widget-convert .cv-out").innerText()).replace(/,/g, ""));
check("converter shows a result", cvFirst > 0);
await page.locator(".widget-convert .cv-amount").fill("2");
await page.waitForTimeout(100);
const cvSecond = parseFloat((await page.locator(".widget-convert .cv-out").innerText()).replace(/,/g, ""));
check("converter recomputes on amount change", cvSecond > cvFirst);

// ---- crypto global bar + trending -----------------------------------------------
await gotoWidget("markets");
await page.waitForSelector(".global-bar");
check("markets global bar renders", (await page.locator(".global-bar .gm-stat").count()) >= 3);
check("markets widget has no stray null text", !(await page.locator(".widget-markets").innerText()).split("\n").includes("null"));
check("fear & greed gauge shows", /\d/.test(await page.locator(".gm-fg-v").innerText()));
check("trending strip renders chips", (await page.locator(".trend-chip").count()) >= 3);

// ---- crypto detail drawer (chart + indicators) ----------------------------------
await gotoWidget("markets");
await page.waitForSelector(".market-row");
await page.locator(".market-row").first().click();
await page.waitForSelector(".detail-pop .coin-price", { timeout: 8000 });
check("coin detail opens with price", /\$[\d,]/.test(await page.locator(".coin-price").innerText()));
check("coin detail renders a candle chart", (await page.locator(".coin-chart-wrap rect").count()) > 10);
check("coin detail shows technical signals", (await page.locator(".coin-signal").count()) >= 3);
check("coin detail shows stat grid", (await page.locator(".coin-stat").count()) >= 6);
// portfolio: enter a holding → value, P/L and allocation donut appear
await page.locator(".hold-input").first().fill("0.5");
await page.locator(".hold-input").nth(1).fill("40000");
await page.locator(".hold-box .btn-primary").click();
await page.waitForSelector(".pf-section .pf-total", { timeout: 5000 });
check("portfolio computes a total value", /\$[\d,]/.test(await page.locator(".pf-total").innerText()));
check("portfolio renders an allocation donut", (await page.locator(".pf-section .chart-donut path").count()) >= 1);
check("holding shows a P/L", /P\/L/.test(await page.locator(".hold-value").innerText()));
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
await gotoWidget("markets");
await page.waitForSelector(".widget-markets .market-row");
const marketCountBefore = await page.locator(".widget-markets .market-row").count();
page.once("dialog", (dialog) => dialog.accept("solana"));
// remove SOL first (edit mode), then re-add it via the prompt
await page.locator("#edit-toggle").click();
await page.waitForSelector(".widget-markets .market-row .icon-btn[title='Remove from watchlist']");
await page.locator(".widget-markets .market-row", { hasText: "SOL" }).locator(".icon-btn[title='Remove from watchlist']").click();
await page.waitForFunction((n) => document.querySelectorAll(".widget-markets .market-row").length === n - 1,
  marketCountBefore, { timeout: 10000 });
check("watchlist remove works", true);
await page.locator(".widget-markets .market-note-row .link-btn").click(); // prompt answered above
await page.waitForFunction((n) => document.querySelectorAll(".widget-markets .market-row").length === n,
  marketCountBefore, { timeout: 10000 });
check("watchlist add works", true);
await page.locator("#edit-toggle").click();

// ---- voice controls (presence + graceful degradation) ----------------------------
await gotoWidget("agent");
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
await gotoWidget("focus");
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
await page.getByRole("button", { name: "Clock", exact: true }).click();
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
  await authPage.waitForSelector(".ws-row", { timeout: 15000 }); // Main page loads worldstate
  check("correct code unlocks and data loads", true);
  await shot(authPage, "08-lockscreen-unlocked");
  await authCtx.close();
} else {
  console.log("  – auth lock-screen checks skipped (set AUTH_URL + AUTH_TOKEN)");
}

// ---- news search (client-side filter, no refetch) ---------------------------
await gotoWidget("news");
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

// ---- legacy single-page state backfills the default pages -----------------------
// Reproduces an early "Main"-only state (pre multi-page defaults): on reload the
// missing default pages must be added so navigation isn't stuck on one tab.
await page.evaluate(() => {
  localStorage.setItem("hermesHub.v1", JSON.stringify({
    version: 1, theme: "dark", pages: [{ id: "legacy-main", name: "Main", layout: [] }],
    activePage: "legacy-main",
  }));
});
await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector(".pagetab", { timeout: 10000 });
const tabNames = await page.$$eval(".pagetab", (els) => els.map((e) => e.textContent.trim()));
check(`legacy Main-only state backfills all pages (${tabNames.length})`,
  ["Markets", "Feeds", "Sports", "Intel", "Health"].every((n) => tabNames.some((t) => t.includes(n))));

// ---- deep-link: ?page= selects a page (powers PWA shortcuts) --------------------
await page.goto(`${BASE}/?page=Intel`, { waitUntil: "networkidle" });
await page.waitForSelector(".pagetab-on", { timeout: 10000 });
check("deep-link ?page=Intel activates the Intel page",
  (await page.locator(".pagetab-on").first().innerText()).trim() === "Intel");
// switching pages keeps the URL in sync (bookmarkable)
await page.locator(".pagetab", { hasText: "Feeds" }).first().click();
await page.waitForFunction(() => new URLSearchParams(location.search).get("page") === "Feeds",
  null, { timeout: 5000 });
check("switching pages updates ?page= in the URL", true);

// ---- tablet / landscape: no horizontal overflow --------------------------------
for (const w of [768, 820, 900]) {
  await page.setViewportSize({ width: w, height: 1024 });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector(".pagetab", { timeout: 10000 });
  const ok = await page.evaluate(() =>
    document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
  check(`tablet ${w}px: no horizontal overflow`, ok);
}

// ---- mobile: page tabs become a pinned bottom nav -------------------------------
await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector(".pagetab", { timeout: 10000 });
await page.evaluate(() => window.scrollTo(0, 1200));   // scroll a long page
await page.waitForTimeout(300);
const navPinned = await page.evaluate(() => {
  const el = document.querySelector(".pagetabs");
  const r = el.getBoundingClientRect();
  return getComputedStyle(el).position === "fixed" && Math.abs(r.bottom - window.innerHeight) < 3;
});
check("mobile: page tabs pinned as bottom nav while scrolled", navPinned);
const noHScroll = await page.evaluate(() =>
  document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check("mobile: no horizontal page overflow", noHScroll);
await page.setViewportSize({ width: 1280, height: 900 });

// ---- console health -----------------------------------------------------------------
const realErrors = errors.filter((e) => !e.includes("favicon"));
check(`no console/page errors (${realErrors.length})`, realErrors.length === 0);
if (realErrors.length) console.log(realErrors.slice(0, 5).map((e) => "    " + e).join("\n"));

await browser.close();
console.log(failures === 0 ? "\nALL E2E CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
