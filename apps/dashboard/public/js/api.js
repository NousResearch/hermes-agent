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

/**
 * POST to an SSE endpoint and read it incrementally. `onDelta(text)` fires
 * for each text chunk; resolves with the final "done" payload (same shape
 * as the non-streaming endpoint).
 */
async function streamJSON(path, body, onDelta) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent("hub:auth-required"));
    throw new Error("access code required");
  }
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "delta") onDelta?.(payload.text);
      else if (event === "done") result = payload;
      else if (event === "error") throw new Error(payload.error);
    }
  }
  if (!result) throw new Error("stream ended without a result");
  return result;
}

export const api = {
  news: (topic, limit = 30) => getJSON("/api/news", { topic, limit }),
  newsAll: (limit = 60) => getJSON("/api/news", { all: 1, limit }),
  weather: (lat, lon, name) =>
    getJSON("/api/weather", name ? { lat, lon, name } : { lat, lon }),
  air: (lat, lon, name) =>
    getJSON("/api/air", name ? { lat, lon, name } : { lat, lon }),
  marine: (lat, lon, name) =>
    getJSON("/api/marine", name ? { lat, lon, name } : { lat, lon }),
  spaceweather: () => getJSON("/api/spaceweather"),
  alerts: (lat, lon, name) =>
    getJSON("/api/alerts", name ? { lat, lon, name } : { lat, lon }),
  flights: (lat, lon, name) =>
    getJSON("/api/flights", name ? { lat, lon, name } : { lat, lon }),
  geocode: (q) => getJSON("/api/geocode", { q }),
  markets: (ids) => getJSON("/api/markets", ids?.length ? { ids: ids.join(",") } : {}),
  feeds: () => getJSON("/api/feeds"),
  feedsOp: (body) => postJSON("/api/feeds", body),
  calendars: () => getJSON("/api/calendars"),
  calendarsOp: (body) => postJSON("/api/calendars", body),
  icsEvents: (days = 60) => getJSON("/api/events", { days }),
  worldstate: () => getJSON("/api/worldstate"),
  reader: (url) => getJSON("/api/reader", { url }),
  health: () => getJSON("/api/health"),
  stateGet: () => getJSON("/api/state"),
  stateRev: () => getJSON("/api/state/rev"),
  statePut: (state, baseRev) => postJSON("/api/state", { state, baseRev }),
  assistantStatus: () => getJSON("/api/assistant/status"),
  routing: () => getJSON("/api/assistant/routing"),
  setRouting: (overrides) => postJSON("/api/assistant/routing", { overrides }),
  telemetry: () => getJSON("/api/assistant/telemetry"),
  recordTelemetry: (event) => postJSON("/api/assistant/telemetry", event),
  killswitch: () => getJSON("/api/killswitch"),
  setKillswitch: (frozen) => postJSON("/api/killswitch", { frozen }),
  evolve: () => getJSON("/api/evolve"),
  evolveReflect: () => postJSON("/api/evolve/reflect", {}),
  evolveProposal: (op, id) => postJSON("/api/evolve/proposal", { op, id }),
  chat: (messages, context) => postJSON("/api/assistant/chat", { messages, context }),
  chatStream: (messages, context, onDelta) =>
    streamJSON("/api/assistant/chat-stream", { messages, context }, onDelta),
  medChatStream: (messages, onDelta) =>
    streamJSON("/api/assistant/medchat-stream", { messages }, onDelta),
  pubmed: (q) => getJSON("/api/pubmed", { q }),
  trials: (q) => getJSON("/api/trials", { q }),
  drug: (q) => getJSON("/api/drug", { q }),
  repos: (window) => getJSON("/api/repos", { window }),
  papers: (cat) => getJSON("/api/papers", { cat }),
  aiNews: (topic) => getJSON("/api/ai-news", { topic }),
  commodities: () => getJSON("/api/commodities"),
  runTool: (name, input) => postJSON("/api/assistant/tool", { name, input }),
  cryptoCoin: (id) => getJSON("/api/crypto/coin", { id }),
  cryptoChart: (id, days) => getJSON("/api/crypto/chart", { id, days }),
  cryptoGlobal: () => getJSON("/api/crypto/global"),
  cryptoTrending: () => getJSON("/api/crypto/trending"),
  scores: (league) => getJSON("/api/scores", { league }),
  standings: (league) => getJSON("/api/standings", { league }),
  teamSchedule: (league, team) => getJSON("/api/team-schedule", { league, team }),
  teamNews: (team) => getJSON("/api/team-news", { team }),
  quakes: () => getJSON("/api/quakes"),
  podcast: (url) => getJSON("/api/podcast", { url }),
  fx: (base, symbols) => getJSON("/api/fx", symbols ? { base, symbols } : { base }),
  convert: () => getJSON("/api/convert"),
  social: (network, sub) => getJSON("/api/social", sub ? { network, sub } : { network }),
  gamingFree: () => getJSON("/api/gaming/free"),
  gamingDeals: () => getJSON("/api/gaming/deals"),
  stocks: (symbols) => getJSON("/api/stocks", symbols ? { symbols } : {}),
  stocksHistory: (symbol) => getJSON("/api/stocks/history", { symbol }),
  backupNow: () => postJSON("/api/backup", {}),
  backupGet: (name) => getJSON("/api/backup/get", { name }),
  backupImport: (snapshot) => postJSON("/api/backup/import", { snapshot }),
  backupRestore: (name) => postJSON("/api/backup/restore", { name }),
  automations: () => getJSON("/api/automations"),
  automationsOp: (body) => postJSON("/api/automations", body),
  notifications: (after) => getJSON("/api/notifications", { after }),
  summarize: (kind, title, content) => postJSON("/api/assistant/summarize", { kind, title, content }),
  briefing: (context) => postJSON("/api/assistant/briefing", { context }),
};
