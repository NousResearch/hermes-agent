// Thin client for the local dashboard API (server.py).

async function getJSON(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(query ? `${path}?${query}` : path);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(`${path}: ${detail}`);
  }
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
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
  assistantStatus: () => getJSON("/api/assistant/status"),
  chat: (messages, context) => postJSON("/api/assistant/chat", { messages, context }),
  summarize: (kind, title, content) => postJSON("/api/assistant/summarize", { kind, title, content }),
  briefing: (context) => postJSON("/api/assistant/briefing", { context }),
};
