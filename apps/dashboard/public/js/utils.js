// Shared helpers: DOM construction, formatting, colors, SVG micro-charts.

/** Hyperscript-style element builder. h('div.card', {onclick}, child, ...) */
export function h(spec, attrs = {}, ...children) {
  const [tag, ...classes] = spec.split(".");
  const el = document.createElement(tag || "div");
  if (classes.length) el.className = classes.join(" ");
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2), value);
    } else if (key === "dataset") {
      Object.assign(el.dataset, value);
    } else if (key === "style" && typeof value === "object") {
      Object.assign(el.style, value);
    } else if (key in el && key !== "list" && key !== "form") {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? "" : value);
    }
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(child));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function uid() {
  return Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-3);
}

const UNITS = [
  ["year", 31536000], ["month", 2592000], ["week", 604800],
  ["day", 86400], ["hour", 3600], ["minute", 60],
];

export function timeAgo(iso) {
  if (!iso) return "";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  for (const [unit, span] of UNITS) {
    if (seconds >= span) {
      const n = Math.floor(seconds / span);
      return `${n} ${unit}${n > 1 ? "s" : ""} ago`;
    }
  }
  return "just now";
}

export function fmtPrice(value) {
  if (value >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (value >= 1) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

// Validated categorical palette (dataviz reference; fixed slot order).
export const SERIES_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"];
export const SERIES_DARK = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];

export function isDark() {
  return document.documentElement.dataset.theme === "dark";
}

/** Deterministic palette slot for a string (used for launcher monograms). */
export function hueFor(text) {
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return hash % SERIES_LIGHT.length;
}

export function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/** Build an inline SVG sparkline (single series, no axes). */
export function sparkline(points, { width = 96, height = 28, stroke = "currentColor" } = {}) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("sparkline");
  if (!points || points.length < 2) return svg;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const pad = 2;
  const step = (width - pad * 2) / (points.length - 1);
  const coords = points.map((p, i) => {
    const x = pad + i * step;
    const y = pad + (1 - (p - min) / span) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  path.setAttribute("points", coords.join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", stroke);
  path.setAttribute("stroke-width", "1.75");
  path.setAttribute("stroke-linejoin", "round");
  path.setAttribute("stroke-linecap", "round");
  svg.append(path);
  return svg;
}

// WMO weather interpretation codes → label + icon glyph.
const WEATHER_CODES = [
  [[0], "Clear", "☀️"],
  [[1], "Mostly clear", "🌤️"],
  [[2], "Partly cloudy", "⛅"],
  [[3], "Overcast", "☁️"],
  [[45, 48], "Fog", "🌫️"],
  [[51, 53, 55, 56, 57], "Drizzle", "🌦️"],
  [[61, 63, 65, 66, 67], "Rain", "🌧️"],
  [[71, 73, 75, 77], "Snow", "🌨️"],
  [[80, 81, 82], "Showers", "🌧️"],
  [[85, 86], "Snow showers", "🌨️"],
  [[95, 96, 99], "Thunderstorm", "⛈️"],
];

export function weatherInfo(code) {
  for (const [codes, label, icon] of WEATHER_CODES) {
    if (codes.includes(code)) return { label, icon };
  }
  return { label: "—", icon: "🌡️" };
}

export function debounce(fn, ms = 250) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

let toastHost;
export function toast(message, kind = "info") {
  if (!toastHost) {
    toastHost = h("div.toast-host", { role: "status", "aria-live": "polite" });
    document.body.append(toastHost);
  }
  const node = h(`div.toast.toast-${kind}`, {}, message);
  toastHost.append(node);
  setTimeout(() => node.classList.add("toast-out"), 2600);
  setTimeout(() => node.remove(), 3000);
}
