// Persistent client state (localStorage), with defaults, export and import.

import { uid } from "./utils.js";

const KEY = "hermesHub.v1";

const w = (type, size) => ({ id: uid(), type, size });

function defaultPages() {
  return [
    { id: uid(), name: "Main", layout: [
      w("glance", "xl"), w("clock", "m"), w("worldstate", "xl"), w("agent", "m"),
      w("weather", "m"), w("launcher", "m"), w("tasks", "m"), w("calendar", "m"),
      w("notes", "m"), w("focus", "s"), w("system", "m"),
    ] },
    { id: uid(), name: "Markets", layout: [w("markets", "l"), w("stocks", "l"), w("commodities", "m")] },
    { id: uid(), name: "Feeds", layout: [
      w("news", "l"), w("reading", "m"), w("socials", "m"), w("gaming", "m"),
      w("podcasts", "m"),
    ] },
    { id: uid(), name: "Sports", layout: [w("scores", "l"), w("racing", "m")] },
    { id: uid(), name: "Intel", layout: [
      w("worldclock", "m"), w("quakes", "m"), w("fx", "m"), w("convert", "m"),
      w("air", "m"), w("marine", "m"), w("space", "m"), w("alerts", "m"), w("flights", "m"),
    ] },
    { id: uid(), name: "Health", layout: [
      w("medbot", "l"), w("pubmed", "m"), w("trials", "m"), w("drug", "m"), w("calc", "m"),
      w("meded", "l"),
    ] },
    { id: uid(), name: "AI Lab", layout: [
      w("aidaily", "xl"), w("ailearn", "l"), w("codelab", "l"),
      w("repos", "m"), w("papers", "m"), w("ainews", "m"), w("snippets", "m"),
      w("changelog", "m"), w("tracker", "m"),
    ] },
  ];
}

function defaultState() {
  const pages = defaultPages();
  return {
    version: 1,
    theme: "dark", // auto | light | dark — dark is the house style
    accent: "cyan", // cyan (default) | amber | green | magenta
    editMode: false,
    pages,
    activePage: pages[0].id,
    pagesSeed: 3, // fresh states already have every default page

    launcher: {
      links: [
        { id: uid(), name: "Gmail", url: "https://mail.google.com" },
        { id: uid(), name: "Calendar", url: "https://calendar.google.com" },
        { id: uid(), name: "YouTube", url: "https://youtube.com" },
        { id: uid(), name: "GitHub", url: "https://github.com" },
        { id: uid(), name: "Reddit", url: "https://reddit.com" },
        { id: uid(), name: "Maps", url: "https://maps.google.com" },
        { id: uid(), name: "Spotify", url: "https://open.spotify.com" },
        { id: uid(), name: "Drive", url: "https://drive.google.com" },
        { id: uid(), name: "Netflix", url: "https://netflix.com" },
        { id: uid(), name: "Wikipedia", url: "https://wikipedia.org" },
        { id: uid(), name: "Amazon", url: "https://amazon.com" },
        { id: uid(), name: "X", url: "https://x.com" },
      ],
    },
    tasks: {
      activeList: "today",
      lists: [
        {
          id: "today",
          name: "Today",
          items: [
            { id: uid(), text: "Review the morning headlines", done: false },
            { id: uid(), text: "Plan the week's priorities", done: false },
          ],
        },
        { id: "groceries", name: "Groceries", items: [] },
        { id: "someday", name: "Someday", items: [] },
      ],
    },
    notes: {
      activeNote: null,
      items: [
        {
          id: uid(),
          text: "Welcome to Hermes Hub\nThis pad autosaves as you type. Use “+ New” for more notes.",
          updated: new Date().toISOString(),
        },
      ],
    },
    calendar: { events: [] },
    agent: { history: [] },
    focus: { running: false, endsAt: null, remainingMs: 25 * 60000, minutes: 25, mode: "focus", sessions: [] },
    // Default location: Durban, SA — used by weather + all Intel widgets
    // (air, marine, alerts, flights) until the user adds/locates their own.
    weather: { locations: [{ name: "Durban", lat: -29.8587, lon: 31.0218 }], active: 0 },
    locSeed: 1,
    news: { topic: "top", muted: [], pinned: [] },
    reading: { items: [] },
    newsRead: {}, // url → timestamp of first open (bounded in markRead)
    markets: { ids: ["bitcoin", "ethereum", "solana", "dogecoin"], holdings: {} },
    sports: { league: "nba", teams: [] },
    socials: { network: "hn", sub: "popular" },
    stocks: { symbols: ["^spx", "^ndq", "^dji", "aapl.us", "msft.us", "eurusd"] },
    podcasts: { feeds: [{ name: "The Changelog", url: "https://changelog.com/podcast/feed" }], active: "https://changelog.com/podcast/feed" },
    search: { engine: "google" },
    medbot: { history: [] },
    pubmed: { query: "South Africa clinical guidelines" },
    trials: { query: "South Africa" },
  };
}

const hasBoard = (s) => s && (Array.isArray(s.pages) || Array.isArray(s.layout));

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultState();
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.version !== 1 || !hasBoard(parsed)) {
      return defaultState();
    }
    migrate(parsed);
    // Merge unknown/missing top-level sections from defaults (forward compat).
    const merged = { ...defaultState(), ...parsed, editMode: false };
    normalizePages(merged);
    return merged;
  } catch {
    return defaultState();
  }
}

/** In-place upgrades for state shapes from older builds. */
function migrate(parsed) {
  if (parsed.weather && !Array.isArray(parsed.weather.locations)) {
    parsed.weather = {
      locations: parsed.weather.location ? [parsed.weather.location] : [],
      active: 0,
    };
  }
  // One-time seed of a default location (Durban) for existing users who never
  // set one, so the Intel widgets have accurate local data out of the box.
  if (parsed.weather && Array.isArray(parsed.weather.locations)
      && parsed.weather.locations.length === 0 && !parsed.locSeed) {
    parsed.weather.locations = [{ name: "Durban", lat: -29.8587, lon: 31.0218 }];
    parsed.weather.active = 0;
  }
  if (parsed.weather) parsed.locSeed = 1;
  // Legacy single-board state → one "Main" page holding it, intact.
  if (Array.isArray(parsed.layout) && !Array.isArray(parsed.pages)) {
    parsed.pages = [{ id: uid(), name: "Main", layout: parsed.layout }];
    parsed.activePage = parsed.pages[0].id;
    delete parsed.layout;
  }
}

/** One-time backfill: add any default pages an older state never had. Guarded
 * by pagesSeed so pages a user deliberately deletes later are NOT re-added.
 * Fixes early single-"Main" states that predate the multi-page defaults. */
function backfillDefaultPages(state) {
  if (state.pagesSeed >= 2) return;
  const have = new Set((state.pages || []).map((p) => (p.name || "").toLowerCase()));
  for (const dp of defaultPages()) {
    if (!have.has(dp.name.toLowerCase())) state.pages.push(dp);
  }
  state.pagesSeed = 2;
}

/** Ensure pages is a non-empty array and activePage points at a real page. */
function normalizePages(state) {
  if (!Array.isArray(state.pages) || !state.pages.length) {
    state.pages = defaultPages();
  }
  backfillDefaultPages(state);
  for (const p of state.pages) {
    if (!p.id) p.id = uid();
    if (!Array.isArray(p.layout)) p.layout = [];
    if (!p.name) p.name = "Page";
  }
  if (!state.pages.some((p) => p.id === state.activePage)) {
    state.activePage = state.pages[0].id;
  }
}

const listeners = new Set();

export const store = {
  state: load(),

  save() {
    localStorage.setItem(KEY, JSON.stringify(this.state));
  },

  /** Mutate state via fn, persist, and notify subscribers with a topic tag. */
  update(fn, topic = "state") {
    fn(this.state);
    this.save();
    for (const listener of listeners) listener(topic, this.state);
  },

  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  exportJSON() {
    return JSON.stringify(this.state, null, 2);
  },

  /** The active page object (never null — falls back to the first page). */
  activePageObj() {
    const pages = this.state.pages || [];
    return pages.find((p) => p.id === this.state.activePage) || pages[0];
  },

  /** The active page's widget layout array. */
  activeLayout() {
    return this.activePageObj()?.layout || [];
  },

  /** Replace the whole state (sync adopting another device's copy). */
  replace(incoming) {
    if (!hasBoard(incoming)) return;
    migrate(incoming);
    this.state = { ...defaultState(), ...incoming, editMode: false };
    normalizePages(this.state);
    this.save();
    for (const listener of listeners) listener("replace", this.state);
  },

  importJSON(text) {
    const incoming = JSON.parse(text); // throws on invalid JSON
    if (!incoming || incoming.version !== 1 || !hasBoard(incoming)) {
      throw new Error("not a Hermes Hub backup file");
    }
    migrate(incoming);
    this.state = { ...defaultState(), ...incoming, editMode: false };
    normalizePages(this.state);
    this.save();
    for (const listener of listeners) listener("import", this.state);
  },

  reset() {
    this.state = defaultState();
    this.save();
    for (const listener of listeners) listener("reset", this.state);
  },
};
