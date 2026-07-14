import { h, clear, uid, toast } from "./utils.js";
import { store } from "./store.js";
import { api } from "./api.js";
import { showLockScreen } from "./auth.js";
import { initSync } from "./sync.js";

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
import worldstate from "./widgets/worldstate.js";

const WIDGETS = Object.fromEntries(
  [clock, worldstate, agent, weather, launcher, news, tasks, notes, calendar, markets]
    .map((w) => [w.type, w]),
);

const SIZES = ["s", "m", "l", "xl"];

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

function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(store.state.theme) + 1) % order.length];
  store.update((state) => { state.theme = next; }, "theme");
  applyTheme();
  toast(`Theme: ${next}`);
  renderTopbar();
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
  if (editMode) {
    controls.prepend(
      h("button.icon-btn", {
        type: "button", title: "Cycle size", "aria-label": `Resize ${spec.title}`,
        onclick: () => {
          store.update((state) => {
            const target = state.layout.find((l) => l.id === item.id);
            target.size = SIZES[(SIZES.indexOf(target.size) + 1) % SIZES.length];
          }, "layout");
          renderGrid();
        },
      }, "⇔"),
      h("button.icon-btn.widget-remove", {
        type: "button", title: "Remove widget", "aria-label": `Remove ${spec.title}`,
        onclick: () => {
          store.update((state) => {
            state.layout = state.layout.filter((l) => l.id !== item.id);
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
        const from = state.layout.findIndex((l) => l.id === draggedId);
        const to = state.layout.findIndex((l) => l.id === item.id);
        const [moved] = state.layout.splice(from, 1);
        state.layout.splice(to, 0, moved);
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
  for (const item of store.state.layout) {
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
                state.layout.push({ id: uid(), type: spec.type, size: spec.defaultSize });
              }, "layout");
              renderGrid();
              toast(`Added ${spec.title}`);
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
      h("button.btn", {
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

function paletteCommands() {
  const commands = [
    { label: "Toggle edit layout", hint: "layout", run: () => document.getElementById("edit-toggle").click() },
    { label: "Switch theme", hint: "appearance", run: cycleTheme },
    { label: "Refresh all widgets", hint: "data", run: () => renderGrid() },
    { label: "Export backup", hint: "data", run: () => document.querySelector(".menu-item").click() },
  ];
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
    const q = input.value.trim().toLowerCase();
    filtered = paletteCommands().filter((c) => c.label.toLowerCase().includes(q));
    selectedIdx = Math.min(selectedIdx, Math.max(0, filtered.length - 1));
    clear(list);
    filtered.slice(0, 12).forEach((cmd, i) => {
      list.append(h("button.palette-item", {
        type: "button",
        role: "option",
        "aria-selected": String(i === selectedIdx),
        class: i === selectedIdx ? "palette-item palette-active" : "palette-item",
        onclick: () => { closePalette(); cmd.run(); },
      }, h("span", {}, cmd.label), h("span.muted.small", {}, cmd.hint)));
    });
    if (!filtered.length) list.append(h("div.muted.palette-empty", {}, "No matching commands"));
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
  applyTheme();
  renderTopbar();
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
