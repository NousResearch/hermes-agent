// Persistent client state (localStorage), with defaults, export and import.

import { uid } from "./utils.js";

const KEY = "hermesHub.v1";

function defaultState() {
  return {
    version: 1,
    theme: "dark", // auto | light | dark — dark is the house style
    editMode: false,
    layout: [
      { id: uid(), type: "clock", size: "m" },
      { id: uid(), type: "worldstate", size: "xl" },
      { id: uid(), type: "agent", size: "m" },
      { id: uid(), type: "weather", size: "m" },
      { id: uid(), type: "launcher", size: "m" },
      { id: uid(), type: "news", size: "l" },
      { id: uid(), type: "tasks", size: "m" },
      { id: uid(), type: "markets", size: "m" },
      { id: uid(), type: "calendar", size: "m" },
      { id: uid(), type: "notes", size: "m" },
    ],
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
    weather: { location: null }, // null → server default until the user picks a city
    news: { topic: "top" },
    markets: { ids: ["bitcoin", "ethereum", "solana", "dogecoin"] },
    search: { engine: "google" },
  };
}

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultState();
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.layout)) {
      return defaultState();
    }
    // Merge unknown/missing top-level sections from defaults (forward compat).
    return { ...defaultState(), ...parsed, editMode: false };
  } catch {
    return defaultState();
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

  /** Replace the whole state (sync adopting another device's copy). */
  replace(incoming) {
    if (!incoming || !Array.isArray(incoming.layout)) return;
    this.state = { ...defaultState(), ...incoming, editMode: false };
    this.save();
    for (const listener of listeners) listener("replace", this.state);
  },

  importJSON(text) {
    const incoming = JSON.parse(text); // throws on invalid JSON
    if (!incoming || incoming.version !== 1 || !Array.isArray(incoming.layout)) {
      throw new Error("not a Hermes Hub backup file");
    }
    this.state = { ...defaultState(), ...incoming, editMode: false };
    this.save();
    for (const listener of listeners) listener("import", this.state);
  },

  reset() {
    this.state = defaultState();
    this.save();
    for (const listener of listeners) listener("reset", this.state);
  },
};
