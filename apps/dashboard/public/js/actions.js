// Executes the agent's dashboard tool calls against local state.
// Tool names/inputs must stay in sync with DASHBOARD_TOOLS in assistant.py.

import { uid } from "./utils.js";
import { store } from "./store.js";
import { openViewer } from "./viewer.js";
import { api } from "./api.js";

// Tools that read server data or manage server-side automations/memory are
// proxied straight through to the server (must mirror assistant.SERVER_TOOLS).
const SERVER_TOOLS = new Set([
  "get_news", "read_article", "get_weather", "get_worldstate", "get_markets",
  "remember", "create_automation", "list_automations", "delete_automation",
]);

function findList(state, name) {
  const lower = (name || "").toLowerCase();
  return state.tasks.lists.find((l) => l.name.toLowerCase() === lower);
}

const HANDLERS = {
  add_task({ text, list }) {
    if (!text) throw new Error("empty task");
    const listName = list || "Today";
    store.update((state) => {
      let target = findList(state, listName);
      if (!target) {
        target = { id: uid(), name: listName, items: [] };
        state.tasks.lists.push(target);
      }
      target.items.push({ id: uid(), text, done: false });
      state.tasks.activeList = target.id;
    }, "tasks-external");
    return `added task “${text}” → ${listName}`;
  },

  complete_task({ text }) {
    const needle = (text || "").toLowerCase();
    let hit = null;
    store.update((state) => {
      for (const list of state.tasks.lists) {
        const item = list.items.find(
          (i) => !i.done && i.text.toLowerCase().includes(needle),
        );
        if (item) {
          item.done = true;
          hit = `${item.text} (${list.name})`;
          state.tasks.activeList = list.id;
          break;
        }
      }
    }, "tasks-external");
    if (!hit) throw new Error(`no open task matching “${text}”`);
    return `completed ${hit}`;
  },

  add_event({ date, title }) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date || "")) throw new Error("date must be YYYY-MM-DD");
    store.update((state) => {
      state.calendar.events.push({ id: uid(), date, title });
    }, "calendar-external");
    return `event “${title}” on ${date}`;
  },

  add_note({ text }) {
    store.update((state) => {
      const id = uid();
      state.notes.items.push({ id, text, updated: new Date().toISOString() });
      state.notes.activeNote = id;
    }, "notes-external");
    return "note created";
  },

  add_app({ name, url }) {
    if (!/^https?:\/\//i.test(url || "")) throw new Error("url must be http(s)");
    store.update((state) => {
      state.launcher.links.push({ id: uid(), name: name || url, url });
    }, "launcher-external");
    return `added ${name} to launcher`;
  },

  open_url({ url, title }) {
    if (!/^https?:\/\//i.test(url || "")) throw new Error("url must be http(s)");
    openViewer({ url, title: title || url, mode: "embed" });
    return `opened ${title || url}`;
  },

  switch_news_topic({ topic }) {
    store.update((state) => {
      state.news.topic = topic;
    }, "news-external");
    return `news topic → ${topic}`;
  },
};

/** Run one tool call. Returns {ok, result} — errors are reported, not thrown. */
export async function executeAction(name, input) {
  if (SERVER_TOOLS.has(name)) {
    try {
      const { result } = await api.runTool(name, input || {});
      return { ok: true, result };
    } catch (err) {
      return { ok: false, result: String(err.message || err) };
    }
  }
  const handler = HANDLERS[name];
  if (!handler) return { ok: false, result: `unknown tool ${name}` };
  try {
    return { ok: true, result: handler(input || {}) };
  } catch (err) {
    return { ok: false, result: String(err.message || err) };
  }
}

/** Snapshot of dashboard state the assistant is allowed to see. */
export function buildContext() {
  const state = store.state;
  return {
    date: new Date().toISOString().slice(0, 10),
    tasks: state.tasks.lists.map((l) => ({
      name: l.name,
      items: l.items.map((i) => ({ text: i.text, done: i.done })),
    })),
    events: state.calendar.events.map((e) => ({ date: e.date, title: e.title })),
    notes: state.notes.items.map((n) => (n.text || "").split("\n")[0].slice(0, 60)),
    apps: state.launcher.links.map((l) => ({ name: l.name, url: l.url })),
  };
}
