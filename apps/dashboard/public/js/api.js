// Thin client for the local dashboard API (server.py).

const TOKEN_KEY = "hermesHub.token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle(res, path) {
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent("hub:auth-required"));
    throw new Error("access code required");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error || detail;
    } catch { /* non-JSON error body */ }
    const err = new Error(`${path}: ${detail}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

async function getJSON(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(query ? `${path}?${query}` : path, { headers: authHeaders() });
  return handle(res, path);
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  return handle(res, path);
}

export const api = {
  news: (topic, limit = 30) => getJSON("/api/news", { topic, limit }),
  weather: (lat, lon, name) =>
    getJSON("/api/weather", name ? { lat, lon, name } : { lat, lon }),
  geocode: (q) => getJSON("/api/geocode", { q }),
  markets: () => getJSON("/api/markets"),
  worldstate: () => getJSON("/api/worldstate"),
  reader: (url) => getJSON("/api/reader", { url }),
  health: () => getJSON("/api/health"),
  stateGet: () => getJSON("/api/state"),
  stateRev: () => getJSON("/api/state/rev"),
  statePut: (state, baseRev) => postJSON("/api/state", { state, baseRev }),
  assistantStatus: () => getJSON("/api/assistant/status"),
  chat: (messages, context) => postJSON("/api/assistant/chat", { messages, context }),
  runTool: (name, input) => postJSON("/api/assistant/tool", { name, input }),
  automations: () => getJSON("/api/automations"),
  automationsOp: (body) => postJSON("/api/automations", body),
  notifications: (after) => getJSON("/api/notifications", { after }),
  summarize: (kind, title, content) => postJSON("/api/assistant/summarize", { kind, title, content }),
  briefing: (context) => postJSON("/api/assistant/briefing", { context }),
};
