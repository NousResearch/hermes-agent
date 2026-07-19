import { h, clear, uid, toast } from "./utils.js";
import { store } from "./store.js";
import { api } from "./api.js";
import { showLockScreen } from "./auth.js";
import { initSync } from "./sync.js";
import { initNotifications } from "./notifications.js";
import { openSources } from "./sources.js";
import { openCalendars } from "./calendars.js";
import { openEvolve } from "./evolve.js";
import { openRouting } from "./routing.js";
import { openDetail } from "./detail.js";

import { openViewer } from "./viewer.js";
import { summarizeButton } from "./summarize.js";
import agent from "./widgets/agent.js";
import clock from "./widgets/clock.js";
import weather from "./widgets/weather.js";
import launcher from "./widgets/launcher.js";
import news from "./widgets/news.js";
import tasks from "./widgets/tasks.js";
import notes from "./widgets/notes.js";
import calendar from "./widgets/calendar.js";
import markets from "./widgets/markets.js";
import scores from "./widgets/scores.js";
import socials from "./widgets/socials.js";
import gaming from "./widgets/gaming.js";
import stocks from "./widgets/stocks.js";
import glance from "./widgets/glance.js";
import worldclock from "./widgets/worldclock.js";
import quakes from "./widgets/quakes.js";
import fx from "./widgets/fx.js";
import convert from "./widgets/convert.js";
import air from "./widgets/air.js";
import space from "./widgets/space.js";
import alerts from "./widgets/alerts.js";
import flights from "./widgets/flights.js";
import podcasts from "./widgets/podcasts.js";
import medbot from "./widgets/medbot.js";
import pubmed from "./widgets/pubmed.js";
import trials from "./widgets/trials.js";
import drug from "./widgets/drug.js";
import calc from "./widgets/calc.js";
import meded from "./widgets/meded.js";
import worldstate from "./widgets/worldstate.js";
import reading from "./widgets/reading.js";
import focus from "./widgets/focus.js";
import system from "./widgets/system.js";

const WIDGETS = Object.fromEntries(
  [clock, glance, worldstate, agent, weather, launcher, news, reading, tasks, notes, calendar, markets, scores, socials, gaming, stocks, worldclock, quakes, fx, convert, air, space, alerts, flights, podcasts, medbot, pubmed, trials, drug, calc, meded, focus, system]
    .map((w) => [w.type, w]),
);

const SIZES = ["s", "m", "l", "xl"];

// The active page object within a state snapshot (mutations target its layout).
const activePage = (state) => state.pages.find((p) => p.id === state.activePage) || state.pages[0];

const SEARCH_ENGINES = {
  google: { name: "Google", url: "https://www.google.com/search?q=" },
  ddg: { name: "DuckDuckGo", url: "https://duckduckgo.com/?q=" },
  youtube: { name: "YouTube", url: "https://www.youtube.com/results?search_query=" },
  wikipedia: { name: "Wikipedia", url: "https://en.wikipedia.org/w/index.php?search=" },
  github: { name: "GitHub", url: "https://github.com/search?q=" },
};

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
function applyTheme() {
  const pref = store.state.theme;
  const dark = pref === "dark" ||
    (pref === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

// Accent presets — each stays within the intelligence-agency aesthetic. "cyan"
// is the house default; picking it clears the overrides so the per-theme tokens
// (which differ light/dark) apply. Others set the three accent vars inline.
const ACCENTS = {
  cyan:   null,
  amber:  { accent: "#f2b13c", dim: "rgba(242,177,60,0.13)",  ink: "#1a1206" },
  green:  { accent: "#35d07f", dim: "rgba(53,208,127,0.13)",  ink: "#04120b" },
  magenta:{ accent: "#e26bd0", dim: "rgba(226,107,208,0.13)", ink: "#1a0616" },
};
const ACCENT_ORDER = Object.keys(ACCENTS);

function applyAccent() {
  const root = document.documentElement;
  const preset = ACCENTS[store.state.accent] ?? null;
  if (!preset) {
    root.style.removeProperty("--accent");
    root.style.removeProperty("--accent-dim");
    root.style.removeProperty("--accent-ink");
    return;
  }
  root.style.setProperty("--accent", preset.accent);
  root.style.setProperty("--accent-dim", preset.dim);
  root.style.setProperty("--accent-ink", preset.ink);
}

function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(store.state.theme) + 1) % order.length];
  store.update((state) => { state.theme = next; }, "theme");
  applyTheme();
  toast(`Theme: ${next}`);
  renderTopbar();
}

function setAccent(name) {
  if (!(name in ACCENTS)) return;
  store.update((state) => { state.accent = name; }, "accent");
  applyAccent();
  renderTopbar();
  toast(`Accent: ${name}`);
}

// ---------------------------------------------------------------------------
// Widget cards
// ---------------------------------------------------------------------------
const cleanups = new Map(); // layout item id → [fns]

function makeCtx(card, itemId, badge) {
  const refreshHandlers = [];
  const storeHandlers = [];
  const timers = [];
  cleanups.set(itemId, [
    () => timers.forEach(clearInterval),
    ...storeHandlers.map((off) => off),
  ]);
  return {
    store,
    api,
    card,
    setBadge(kind) {
      badge.textContent = kind === "sample" ? "demo data" : "";
      badge.title = kind === "sample"
        ? "Live feeds unreachable — showing bundled sample data" : "";
      badge.hidden = !kind;
    },
    onRefresh(fn) { refreshHandlers.push(fn); },
    triggerRefresh() { refreshHandlers.forEach((fn) => fn()); },
    onStore(fn) { storeHandlers.push(store.subscribe(fn)); },
    every(ms, fn) { timers.push(setInterval(fn, ms)); },
    onSummarize(getPayload) { this._summarize = getPayload; },
  };
}

function renderCard(item) {
  const spec = WIDGETS[item.type];
  if (!spec) return null;
  const editMode = store.state.editMode;

  const badge = h("span.widget-badge", { hidden: true });
  const body = h("div.widget-body");
  const card = h("section.widget", {
    class: `widget widget-${item.type} size-${item.size}`,
    dataset: { id: item.id, type: item.type },
    "aria-label": spec.title,
    draggable: editMode,
  });

  const controls = h("div.widget-controls", {});
  const ctx = makeCtx(card, item.id, badge);

  controls.append(
    h("button.icon-btn", {
      type: "button", title: "Refresh", "aria-label": `Refresh ${spec.title}`,
      onclick: () => ctx.triggerRefresh(),
    }, "↻"),
  );
  // Widgets that export a detail() renderer get an expand-to-window control.
  if (typeof spec.detail === "function") {
    controls.append(
      h("button.icon-btn.widget-expand", {
        type: "button", title: "Expand", "aria-label": `Expand ${spec.title}`,
        onclick: () => openDetail(spec, ctx),
      }, "⤢"),
    );
  }
  if (editMode) {
    controls.prepend(
      h("button.icon-btn", {
        type: "button", title: "Cycle size", "aria-label": `Resize ${spec.title}`,
        onclick: () => {
          store.update((state) => {
            const target = activePage(state).layout.find((l) => l.id === item.id);
            target.size = SIZES[(SIZES.indexOf(target.size) + 1) % SIZES.length];
          }, "layout");
          renderGrid();
        },
      }, "⇔"),
      h("button.icon-btn.widget-remove", {
        type: "button", title: "Remove widget", "aria-label": `Remove ${spec.title}`,
        onclick: () => {
          store.update((state) => {
            const page = activePage(state);
            page.layout = page.layout.filter((l) => l.id !== item.id);
          }, "layout");
          renderGrid();
        },
      }, "✕"),
    );
  }

  card.append(
    h("header.widget-head", {},
      h("span.widget-icon", { "aria-hidden": "true" }, spec.icon),
      h("h2.widget-title", {}, spec.title),
      badge,
      editMode ? h("span.widget-grip", { title: "Drag to reorder", "aria-hidden": "true" }, "⠿") : null,
      controls,
    ),
    body,
  );

  // Card head is bound before render so widgets can set badges immediately.
  card._ctx = ctx;
  spec.render(body, ctx);
  if (ctx._summarize) {
    controls.prepend(summarizeButton(ctx._summarize, {
      tip: `Summarize ${spec.title} with AI`,
    }));
  }

  if (editMode) {
    card.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/widget-id", item.id);
      ev.dataTransfer.effectAllowed = "move";
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
    card.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      card.classList.add("drop-target");
    });
    card.addEventListener("dragleave", () => card.classList.remove("drop-target"));
    card.addEventListener("drop", (ev) => {
      ev.preventDefault();
      card.classList.remove("drop-target");
      const draggedId = ev.dataTransfer.getData("text/widget-id");
      if (!draggedId || draggedId === item.id) return;
      store.update((state) => {
        const layout = activePage(state).layout;
        const from = layout.findIndex((l) => l.id === draggedId);
        const to = layout.findIndex((l) => l.id === item.id);
        const [moved] = layout.splice(from, 1);
        layout.splice(to, 0, moved);
      }, "layout");
      renderGrid();
    });
  }
  return card;
}

// Masonry: cards span N implicit 10px rows based on their content height, so
// columns pack tightly with no dead vertical space between cards.
const MASONRY_ROW = 10;
const MASONRY_GAP = 14;
let masonryObserver = null;

function fitCard(card) {
  card.style.gridRowEnd = "span 1";
  const height = card.scrollHeight + 2; // + borders
  card.style.gridRowEnd = `span ${Math.max(4, Math.ceil((height + MASONRY_GAP) / (MASONRY_ROW + MASONRY_GAP)))}`;
}

function renderGrid() {
  const grid = document.getElementById("grid");
  for (const fns of cleanups.values()) fns.forEach((fn) => fn());
  cleanups.clear();
  masonryObserver?.disconnect();
  clear(grid);
  const cards = [];
  for (const item of store.activeLayout()) {
    const card = renderCard(item);
    if (card) { grid.append(card); cards.push(card); }
  }
  if (store.state.editMode) {
    const gallery = renderAddGallery();
    grid.append(gallery);
    cards.push(gallery);
  }
  document.body.classList.toggle("edit-mode", store.state.editMode);

  masonryObserver = new ResizeObserver((entries) => {
    for (const entry of entries) fitCard(entry.target.closest(".widget"));
  });
  for (const card of cards) {
    fitCard(card);
    masonryObserver.observe(card.querySelector(".widget-body"));
  }
}
window.addEventListener("resize", () => {
  document.querySelectorAll(".widget").forEach(fitCard);
});

function switchPage(id) {
  if (store.state.activePage === id) return;
  store.update((s) => { s.activePage = id; }, "activePage");
  syncPageUrl(id);
  renderPageTabs();
  renderGrid();
}

// Keep ?page=<name> in the URL so pages are bookmarkable/shareable and PWA
// shortcuts (see manifest) deep-link straight to a page.
function syncPageUrl(id) {
  const page = store.state.pages.find((p) => p.id === id);
  if (!page) return;
  const url = new URL(location.href);
  url.searchParams.set("page", page.name);
  history.replaceState(null, "", url);
}

// On load, honour ?page=<name> (case-insensitive) if it names a real page.
function applyDeepLink() {
  const want = new URLSearchParams(location.search).get("page");
  if (!want) return;
  const page = store.state.pages.find((p) => p.name.toLowerCase() === want.toLowerCase());
  if (page && page.id !== store.state.activePage) {
    store.update((s) => { s.activePage = page.id; }, "activePage");
  }
}

function renderPageTabs() {
  const nav = document.getElementById("pagetabs");
  if (!nav) return;
  const editing = store.state.editMode;
  clear(nav);
  for (const page of store.state.pages) {
    const activeTab = page.id === store.state.activePage;
    nav.append(h("button.pagetab", {
      type: "button",
      "aria-current": activeTab ? "page" : null,
      class: activeTab ? "pagetab pagetab-on" : "pagetab",
      ondblclick: editing ? () => renamePage(page) : null,
      onclick: () => switchPage(page.id),
    },
      h("span", {}, page.name),
      editing && store.state.pages.length > 1
        ? h("span.pagetab-x", {
            title: "Delete page", "aria-label": `Delete page ${page.name}`,
            onclick: (ev) => { ev.stopPropagation(); deletePage(page); },
          }, "✕")
        : null,
    ));
  }
  if (editing) {
    nav.append(h("button.pagetab.pagetab-add", {
      type: "button", title: "Add a page",
      onclick: () => {
        const name = prompt("Name for the new page:");
        if (!name?.trim()) return;
        const id = uid();
        store.update((s) => { s.pages.push({ id, name: name.trim(), layout: [] }); s.activePage = id; }, "pages");
        renderPageTabs();
        renderGrid();
      },
    }, "＋ Page"));
    nav.append(h("span.pagetab-hint.muted.small", {}, "double-click a tab to rename"));
  }
}

function renamePage(page) {
  const name = prompt("Rename page:", page.name);
  if (!name?.trim()) return;
  store.update((s) => {
    const p = s.pages.find((x) => x.id === page.id);
    if (p) p.name = name.trim();
  }, "pages");
  renderPageTabs();
}

function deletePage(page) {
  if (store.state.pages.length <= 1) return;
  if (page.layout.length && !confirm(`Delete “${page.name}” and its ${page.layout.length} widget(s)?`)) return;
  store.update((s) => {
    s.pages = s.pages.filter((p) => p.id !== page.id);
    if (s.activePage === page.id) s.activePage = s.pages[0].id;
  }, "pages");
  renderPageTabs();
  renderGrid();
}

function renderAddGallery() {
  const missing = Object.values(WIDGETS);
  return h("section.widget.add-gallery.size-m", { "aria-label": "Add widgets" },
    h("header.widget-head", {}, h("h2.widget-title", {}, "Add a widget")),
    h("div.widget-body", {},
      h("div.gallery-grid", {},
        missing.map((spec) =>
          h("button.gallery-item", {
            type: "button",
            onclick: () => {
              store.update((state) => {
                activePage(state).layout.push({ id: uid(), type: spec.type, size: spec.defaultSize });
              }, "layout");
              renderGrid();
              toast(`Added ${spec.title} to ${activePage(store.state).name}`);
            },
          }, h("span", { "aria-hidden": "true" }, spec.icon), ` ${spec.title}`),
        ),
      ),
      h("p.muted.small", {}, "Widgets can be added multiple times, dragged to reorder, resized (⇔) and removed (✕)."),
    ),
  );
}

// ---------------------------------------------------------------------------
// Top bar: search, actions
// ---------------------------------------------------------------------------
function runSearch(query) {
  let engine = store.state.search.engine;
  let text = query.trim();
  // Bang-style prefix: "yt cats", "gh vite", "w helios", "ddg foo", "g bar"
  const prefixes = { g: "google", ddg: "ddg", yt: "youtube", w: "wikipedia", gh: "github" };
  const match = text.match(/^(\w{1,3})\s+(.*)$/);
  if (match && prefixes[match[1].toLowerCase()]) {
    engine = prefixes[match[1].toLowerCase()];
    text = match[2];
  }
  if (!text) return;
  if (/^https?:\/\/\S+$/i.test(text)) {
    window.open(text, "_blank", "noopener");
    return;
  }
  window.open(SEARCH_ENGINES[engine].url + encodeURIComponent(text), "_blank", "noopener");
}

function renderTopbar() {
  const bar = document.getElementById("topbar");
  clear(bar);

  const input = h("input.search-input", {
    id: "search",
    type: "search",
    placeholder: "Search the web…  (g/ddg/yt/w/gh prefix picks the engine, ⌘K for commands)",
    "aria-label": "Search the web",
  });
  const engineSelect = h("select.select.search-engine", {
    "aria-label": "Search engine",
    onchange: (ev) => store.update((state) => { state.search.engine = ev.target.value; }, "search"),
  },
    Object.entries(SEARCH_ENGINES).map(([key, eng]) =>
      h("option", { value: key, selected: key === store.state.search.engine }, eng.name)),
  );
  const form = h("form.search-form", {
    role: "search",
    onsubmit: (ev) => { ev.preventDefault(); runSearch(input.value); input.select(); },
  }, engineSelect, input);

  const themeLabel = { auto: "◐", light: "☀", dark: "☾" }[store.state.theme];

  bar.append(
    h("div.brand", {},
      h("span.brand-mark", { "aria-hidden": "true" }, "◆"),
      h("span.brand-name", {}, "HERMES", h("span.brand-sub", {}, "//HUB")),
      h("span.brand-tag", {}, "LOCAL · EYES ONLY"),
    ),
    form,
    h("div.topbar-actions", {},
      h("button.btn.palette-btn", {
        type: "button", title: "Command palette (Ctrl/⌘+K)",
        onclick: openPalette,
      }, "⌘K"),
      h("button.btn", {
        type: "button",
        title: `Theme: ${store.state.theme} (click to change)`,
        "aria-label": `Theme: ${store.state.theme}`,
        onclick: cycleTheme,
      }, themeLabel),
      h("button.btn", {
        type: "button",
        id: "edit-toggle",
        class: store.state.editMode ? "btn btn-primary" : "btn",
        "aria-pressed": String(store.state.editMode),
        onclick: () => {
          store.update((state) => { state.editMode = !state.editMode; }, "editMode");
          renderTopbar();
          renderPageTabs();
          renderGrid();
        },
      }, store.state.editMode ? "Done" : "Edit layout"),
      renderSettingsMenu(),
    ),
  );
}

function renderSettingsMenu() {
  const menu = h("div.menu", { hidden: true },
    h("button.menu-item", {
      type: "button",
      onclick: () => openSources(),
    }, "News sources…"),
    h("button.menu-item", {
      type: "button",
      onclick: () => openCalendars(),
    }, "Calendar feeds…"),
    h("button.menu-item", {
      type: "button",
      onclick: () => openEvolve(),
    }, "Agent proposals…"),
    h("button.menu-item", {
      type: "button",
      onclick: () => openRouting(),
    }, "Model routing…"),
    h("div.menu-accents", { role: "group", "aria-label": "Accent color" },
      h("span.menu-accents-label", {}, "ACCENT"),
      ...ACCENT_ORDER.map((name) =>
        h("button.accent-swatch", {
          type: "button",
          title: name,
          "aria-label": `Accent ${name}`,
          "aria-pressed": String(store.state.accent === name),
          class: store.state.accent === name ? "accent-swatch accent-swatch-on" : "accent-swatch",
          style: { background: ACCENTS[name] ? ACCENTS[name].accent : "#41d3ea" },
          onclick: (ev) => { ev.stopPropagation(); setAccent(name); },
        }),
      ),
    ),
    h("button.menu-item", {
      type: "button",
      onclick: () => {
        const blob = new Blob([store.exportJSON()], { type: "application/json" });
        const a = h("a", {
          href: URL.createObjectURL(blob),
          download: `hermes-hub-backup-${new Date().toISOString().slice(0, 10)}.json`,
        });
        a.click();
        URL.revokeObjectURL(a.href);
        toast("Backup downloaded");
      },
    }, "Export data…"),
    h("button.menu-item", {
      type: "button",
      onclick: () => {
        const picker = h("input", { type: "file", accept: "application/json" });
        picker.addEventListener("change", async () => {
          try {
            store.importJSON(await picker.files[0].text());
            toast("Backup restored");
            boot();
          } catch (err) {
            toast(`Import failed: ${err.message}`, "error");
          }
        });
        picker.click();
      },
    }, "Import data…"),
    h("button.menu-item", {
      type: "button",
      title: "Download a full server-side snapshot (state, feeds, calendars, automations, memory) to keep off-box",
      onclick: async () => {
        try {
          const { name } = await api.backupNow();
          const snap = await api.backupGet(name);
          const blob = new Blob([JSON.stringify(snap)], { type: "application/json" });
          const a = h("a", { href: URL.createObjectURL(blob), download: name });
          a.click();
          URL.revokeObjectURL(a.href);
          toast("Server backup downloaded");
        } catch (err) { toast(`Backup failed: ${err.message}`, "error"); }
      },
    }, "Download server backup…"),
    h("button.menu-item", {
      type: "button",
      title: "Import a downloaded server backup and restore the whole hub from it",
      onclick: () => {
        const picker = h("input", { type: "file", accept: "application/json" });
        picker.addEventListener("change", async () => {
          try {
            const snapshot = JSON.parse(await picker.files[0].text());
            const { name } = await api.backupImport(snapshot);
            await api.backupRestore(name);
            toast("Server backup restored");
            location.reload();
          } catch (err) { toast(`Restore failed: ${err.message}`, "error"); }
        });
        picker.click();
      },
    }, "Restore server backup…"),
    h("button.menu-item.menu-danger", {
      type: "button",
      onclick: () => {
        if (!confirm("Reset the whole dashboard to defaults? Your lists, notes, events and apps will be erased.")) return;
        store.reset();
        toast("Dashboard reset");
        boot();
      },
    }, "Reset everything"),
  );
  const wrap = h("div.menu-wrap", {},
    h("button.btn", {
      type: "button", "aria-haspopup": "true", "aria-label": "Settings",
      onclick: (ev) => { ev.stopPropagation(); menu.hidden = !menu.hidden; },
    }, "⚙"),
    menu,
  );
  document.addEventListener("click", () => { menu.hidden = true; });
  return wrap;
}

// ---------------------------------------------------------------------------
// Command palette
// ---------------------------------------------------------------------------
let paletteEl = null;

/** Scroll a widget into view and pulse it — used when jumping to a search hit. */
function flashWidget(type) {
  // jump to the page holding this widget first, so cross-page search lands
  const page = store.state.pages.find((p) => p.layout.some((wgt) => wgt.type === type));
  if (page && page.id !== store.state.activePage) switchPage(page.id);
  const el = document.querySelector(`.widget-${type}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("widget-flash");
  void el.offsetWidth; // restart the animation if already flashing
  el.classList.add("widget-flash");
  setTimeout(() => el.classList.remove("widget-flash"), 1400);
}

/** Search across the user's own data (tasks, notes, events, reading, apps). */
function paletteDataMatches(q) {
  const matches = [];
  const state = store.state;
  for (const list of state.tasks.lists) {
    for (const item of list.items) {
      if (item.text.toLowerCase().includes(q)) {
        matches.push({
          label: item.text, hint: `task · ${list.name}${item.done ? " ✓" : ""}`,
          run: () => {
            store.update((s) => { s.tasks.activeList = list.id; }, "tasks-external");
            flashWidget("tasks");
          },
        });
      }
    }
  }
  for (const note of state.notes.items) {
    const title = (note.text || "").split("\n")[0].slice(0, 60) || "(untitled note)";
    if ((note.text || "").toLowerCase().includes(q)) {
      matches.push({
        label: title, hint: "note",
        run: () => {
          store.update((s) => { s.notes.activeNote = note.id; }, "notes-external");
          flashWidget("notes");
        },
      });
    }
  }
  for (const ev of state.calendar.events) {
    if ((ev.title || "").toLowerCase().includes(q)) {
      matches.push({ label: ev.title, hint: `event · ${ev.date}`, run: () => flashWidget("calendar") });
    }
  }
  for (const item of state.reading.items || []) {
    if ((item.title || "").toLowerCase().includes(q)) {
      matches.push({
        label: item.title, hint: "reading",
        run: () => openViewer({ url: item.url, title: item.title, mode: "reader" }),
      });
    }
  }
  return matches.slice(0, 8);
}

// Run typed text as an agent command by driving the Agent widget's own form,
// so it reuses the entire agent loop (local parser or Claude) and its
// permission gate — the palette just hands off the text.
function runAgentCommand(text) {
  // make sure the Agent widget is on-screen (it may live on another page)
  const page = store.state.pages.find((p) => p.layout.some((wgt) => wgt.type === "agent"));
  if (page && page.id !== store.state.activePage) switchPage(page.id);
  const input = document.querySelector(".agent-input");
  const submit = document.querySelector(".agent-form .btn-primary");
  if (!input || !submit) {
    toast("Add the Agent widget to run commands", "error");
    return;
  }
  flashWidget("agent");
  input.value = text;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  submit.click();
}

function paletteCommands(query = "") {
  const commands = [
    { label: "Toggle edit layout", hint: "layout", run: () => document.getElementById("edit-toggle").click() },
    { label: "Switch theme", hint: "appearance", run: cycleTheme },
    { label: "Refresh all widgets", hint: "data", run: () => renderGrid() },
    {
      label: "Export backup", hint: "data",
      run: () => [...document.querySelectorAll(".menu-item")]
        .find((el) => el.textContent.startsWith("Export"))?.click(),
    },
    { label: "Manage news sources", hint: "feeds", run: () => openSources() },
  ];
  for (const page of store.state.pages) {
    if (page.id !== store.state.activePage) {
      commands.push({ label: `Go to ${page.name}`, hint: "page", run: () => switchPage(page.id) });
    }
  }
  for (const link of store.state.launcher.links) {
    commands.push({
      label: `Open ${link.name}`,
      hint: new URL(link.url).hostname,
      run: () => openViewer({ url: link.url, title: link.name, mode: "embed" }),
    });
  }
  for (const [key, engine] of Object.entries(SEARCH_ENGINES)) {
    commands.push({
      label: `Search with ${engine.name}`,
      hint: "search",
      run: () => {
        store.update((state) => { state.search.engine = key; }, "search");
        renderTopbar();
        document.getElementById("search").focus();
      },
    });
  }
  // Data hits are query-specific; they lead the list so "find my stuff" is fast.
  // After them, an explicit "run this as a command" hands the text to the agent.
  const q = query.trim().toLowerCase();
  if (q) {
    const runCmd = {
      label: `Run “${query.trim()}” as a command`,
      hint: "agent",
      keep: true, // survive the label-substring filter in refresh()
      run: () => runAgentCommand(query.trim()),
    };
    return [...paletteDataMatches(q), runCmd, ...commands];
  }
  return commands;
}

function openPalette() {
  if (paletteEl) return;
  const input = h("input.palette-input", {
    type: "text", placeholder: "Type a command or app name…",
    "aria-label": "Command palette",
  });
  const list = h("div.palette-list", { role: "listbox" });
  const backdrop = h("div.palette-backdrop", {
    onclick: (ev) => { if (ev.target === backdrop) closePalette(); },
  }, h("div.palette", { role: "dialog", "aria-label": "Command palette" }, input, list));

  let filtered = [];
  let selectedIdx = 0;

  const refresh = () => {
    // Pass the original-case text to paletteCommands (so a run-command keeps
    // the user's casing); lowercase only for the label-substring filter.
    const raw = input.value.trim();
    const q = raw.toLowerCase();
    filtered = paletteCommands(raw).filter((c) => c.keep || c.label.toLowerCase().includes(q));
    selectedIdx = Math.min(selectedIdx, Math.max(0, filtered.length - 1));
    clear(list);
    filtered.slice(0, 12).forEach((cmd, i) => {
      list.append(h("button.palette-item", {
        type: "button",
        role: "option",
        "aria-selected": String(i === selectedIdx),
        class: i === selectedIdx ? "palette-item palette-active" : "palette-item",
        onclick: () => { closePalette(); cmd.run(); },
      }, h("span.palette-label", {}, cmd.label), h("span.muted.small", {}, cmd.hint)));
    });
    if (!filtered.length) list.append(h("div.muted.palette-empty", {}, "No matches"));
  };

  input.addEventListener("input", () => { selectedIdx = 0; refresh(); });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown") { ev.preventDefault(); selectedIdx = Math.min(selectedIdx + 1, filtered.length - 1); refresh(); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); selectedIdx = Math.max(selectedIdx - 1, 0); refresh(); }
    else if (ev.key === "Enter") { ev.preventDefault(); const cmd = filtered[selectedIdx]; if (cmd) { closePalette(); cmd.run(); } }
    else if (ev.key === "Escape") closePalette();
  });

  document.body.append(backdrop);
  paletteEl = backdrop;
  refresh();
  input.focus();
}

function closePalette() {
  paletteEl?.remove();
  paletteEl = null;
}

// ---------------------------------------------------------------------------
// Footer + boot
// ---------------------------------------------------------------------------
async function renderFooter() {
  const footer = document.getElementById("footer");
  clear(footer).append(
    h("span.muted", {}, "HERMES//HUB — all sources, one console."),
    h("span.muted", { id: "conn-status" }, "CHANNEL: PROBING…"),
  );
  try {
    const health = await api.health();
    document.getElementById("conn-status").textContent = health.offline
      ? "CHANNEL: OFFLINE — BUNDLED SAMPLE DATA"
      : "CHANNEL: LIVE — AUTO-REFRESH ENABLED";
  } catch {
    document.getElementById("conn-status").textContent = "CHANNEL: SERVER UNREACHABLE";
  }
}

function bindShortcuts() {
  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      paletteEl ? closePalette() : openPalette();
    } else if (ev.key === "/" && document.activeElement === document.body) {
      ev.preventDefault();
      document.getElementById("search")?.focus();
    }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
}

function boot() {
  applyDeepLink();   // ?page= wins over persisted/synced activePage on load
  applyTheme();
  applyAccent();
  renderTopbar();
  renderPageTabs();
  renderGrid();
  renderFooter();
}

window.addEventListener("hub:auth-required", () => {
  showLockScreen({
    onUnlocked: () => {
      boot();
      initSync(boot);
    },
  });
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => { /* http or old browser */ });
}

bindShortcuts();
boot();
initSync(boot);
initNotifications();
