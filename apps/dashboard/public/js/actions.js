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

// Permission gate (Jarvis Layer F) — mirrors assistant.TOOL_TIERS. The server
// is the source of truth; this copy drives pre-flight UX so the loop doesn't
// have to await a fetch per tool. Unknown tools default to "blocked".
const TOOL_TIERS = {
  add_task: "auto", complete_task: "auto", add_event: "auto", add_note: "auto",
  switch_news_topic: "auto", remember: "auto", get_news: "auto",
  read_article: "auto", get_weather: "auto", get_worldstate: "auto",
  get_markets: "auto", list_automations: "auto",
  add_app: "confirm", open_url: "confirm",
  create_automation: "confirm", delete_automation: "confirm",
};

export function toolTier(name) {
  return TOOL_TIERS[name] || "blocked";
}

/** Human-readable one-liner describing a pending tool call, for approval cards. */
export function describeAction(name, input = {}) {
  switch (name) {
    case "add_app": return `Add app “${input.name || input.url}” → ${input.url}`;
    case "open_url": return `Open ${input.title || input.url} in the viewer`;
    case "create_automation": return `Arm automation “${input.name || "untitled"}”`;
    case "delete_automation": return `Delete automation #${input.id}`;
    default: return `${name} ${JSON.stringify(input)}`;
  }
}

function findList(state, name) {
  const lower = (name || "").toLowerCase();
  return state.tasks.lists.find((l) => l.name.toLowerCase() === lower);
}

const HANDLERS = {
  add_task({ text, list, due, priority }) {
    if (!text) throw new Error("empty task");
    const listName = list || "Today";
    const item = { id: uid(), text, done: false };
    if (due && /^\d{4}-\d{2}-\d{2}$/.test(due)) item.due = due;
    if (["high", "normal", "low"].includes(priority)) item.priority = priority;
    store.update((state) => {
      let target = findList(state, listName);
      if (!target) {
        target = { id: uid(), name: listName, items: [] };
        state.tasks.lists.push(target);
      }
      target.items.push(item);
      state.tasks.activeList = target.id;
    }, "tasks-external");
    const extra = [item.due && `due ${item.due}`, item.priority && `${item.priority} priority`]
      .filter(Boolean).join(", ");
    return `added task “${text}” → ${listName}${extra ? ` (${extra})` : ""}`;
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
    const clean = (topic || "").toLowerCase().trim();
    if (!clean) throw new Error("empty topic");
    store.update((state) => {
      state.news.topic = clean;
    }, "news-external");
    return `news topic → ${clean}`;
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
