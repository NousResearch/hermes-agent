import { h, clear, weatherInfo, debounce, toast } from "../utils.js";

/**
 * 24-hour temperature line chart (SVG): single series, 2px line, hairline
 * grid, crosshair + tooltip on hover. Precipitation gets its own row of
 * dots below the axis rather than a second y-scale.
 */
function tempChart(hours) {
  const W = 560;
  const H = 150;
  const PAD = { top: 14, right: 10, bottom: 26, left: 34 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const temps = hours.map((hr) => hr.temp);
  let min = Math.min(...temps);
  let max = Math.max(...temps);
  if (max - min < 4) { max += 2; min -= 2; } // avoid a flat-looking scale
  const x = (i) => PAD.left + (i / (hours.length - 1)) * plotW;
  const y = (t) => PAD.top + (1 - (t - min) / (max - min)) * plotH;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "weather-chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Temperature for the next 24 hours");

  const make = (tag, attrs) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    svg.append(el);
    return el;
  };

  // hairline gridlines + y labels (3 ticks)
  for (const frac of [0, 0.5, 1]) {
    const value = min + frac * (max - min);
    const gy = y(value);
    make("line", { x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy, class: "chart-grid" });
    const label = make("text", { x: PAD.left - 6, y: gy + 3.5, class: "chart-label", "text-anchor": "end" });
    label.textContent = `${Math.round(value)}°`;
  }
  // x labels every 6 hours
  hours.forEach((hr, i) => {
    if (i % 6 !== 0) return;
    const label = make("text", { x: x(i), y: H - 8, class: "chart-label", "text-anchor": "middle" });
    label.textContent = new Date(hr.time).toLocaleTimeString(undefined, { hour: "numeric" });
  });

  // area fill + line
  const linePts = hours.map((hr, i) => `${x(i).toFixed(1)},${y(hr.temp).toFixed(1)}`);
  make("polygon", {
    points: `${PAD.left},${PAD.top + plotH} ${linePts.join(" ")} ${x(hours.length - 1)},${PAD.top + plotH}`,
    class: "chart-area",
  });
  make("polyline", { points: linePts.join(" "), class: "chart-line", fill: "none" });

  // precipitation dots along the baseline (own encoding, no second axis)
  hours.forEach((hr, i) => {
    if ((hr.precipProb || 0) < 30) return;
    make("circle", {
      cx: x(i), cy: PAD.top + plotH + 6, r: 2.5,
      class: "chart-precip-dot",
      opacity: Math.min(1, hr.precipProb / 100 + 0.25).toFixed(2),
    });
  });

  // hover layer: crosshair + tooltip
  const crosshair = make("line", { y1: PAD.top, y2: PAD.top + plotH, class: "chart-crosshair", visibility: "hidden" });
  const dot = make("circle", { r: 4, class: "chart-hover-dot", visibility: "hidden" });
  const tip = h("div.chart-tooltip", { hidden: true });
  const wrap = h("div.weather-chart-wrap", {}, svg, tip);

  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(hours.length - 1, Math.round(((px - PAD.left) / plotW) * (hours.length - 1))));
    const hr = hours[i];
    crosshair.setAttribute("x1", x(i));
    crosshair.setAttribute("x2", x(i));
    crosshair.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i));
    dot.setAttribute("cy", y(hr.temp));
    dot.setAttribute("visibility", "visible");
    const when = new Date(hr.time).toLocaleTimeString(undefined, { hour: "numeric" });
    tip.textContent = `${when} · ${Math.round(hr.temp)}° · ${hr.precipProb ?? 0}% rain`;
    tip.hidden = false;
    const left = Math.min(Math.max((x(i) / W) * 100, 12), 88);
    tip.style.left = `${left}%`;
  });
  svg.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  return wrap;
}

function citySearch(ctx, onPicked) {
  const input = h("input.input", {
    type: "search",
    placeholder: "Search city…",
    "aria-label": "Search city",
  });
  const results = h("div.city-results");

  // "Use my location" — browser geolocation, no external reverse-geocode.
  const locateBtn = h("button.btn.btn-tiny.city-locate", {
    type: "button", title: "Use my current location",
  }, "📍 Use my location");
  locateBtn.addEventListener("click", () => {
    if (!navigator.geolocation) {
      clear(results).append(h("div.muted", {}, "Geolocation not supported."));
      return;
    }
    locateBtn.disabled = true;
    locateBtn.textContent = "📍 Locating…";
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        onPicked({ name: "My location", lat: Math.round(latitude * 1e4) / 1e4,
          lon: Math.round(longitude * 1e4) / 1e4 });
      },
      (err) => {
        locateBtn.disabled = false;
        locateBtn.textContent = "📍 Use my location";
        clear(results).append(h("div.muted", {},
          err.code === err.PERMISSION_DENIED ? "Location permission denied." : "Couldn't get location."));
      },
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 600000 },
    );
  });
  const runSearch = debounce(async () => {
    const q = input.value.trim();
    clear(results);
    if (q.length < 2) return;
    try {
      const { results: found } = await ctx.api.geocode(q);
      for (const place of found) {
        results.append(
          h("button.city-option", {
            type: "button",
            onclick: () => onPicked(place),
          }, `${place.name}${place.region ? ", " + place.region : ""} — ${place.country}`),
        );
      }
      if (!found.length) results.append(h("div.muted", {}, "No matches"));
    } catch {
      results.append(h("div.muted", {}, "Search unavailable"));
    }
  }, 300);
  input.addEventListener("input", runSearch);
  return h("div.city-search", {}, input, locateBtn, results);
}

export default {
  type: "weather",
  title: "Weather",
  icon: "⛅",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    let showingSearch = false;

    const locations = () => store.state.weather.locations || [];
    const activeLoc = () => locations()[store.state.weather.active] || null;

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "Loading forecast…"));
      const loc = activeLoc();
      let data;
      try {
        data = loc
          ? await ctx.api.weather(loc.lat, loc.lon, loc.name)
          : await ctx.api.weather(-29.8587, 31.0218, "Durban");
      } catch (err) {
        clear(body).append(h("div.widget-error", {}, `Weather unavailable: ${err.message}`));
        return;
      }
      ctx.setBadge(data.source === "sample" ? "sample" : null);
      ctx._track?.(data);
      const { icon, label } = weatherInfo(data.current.code);

      // city tabs (shown once the user has saved at least one city)
      const cityTabs = h("div.note-tabs.weather-cities", { role: "tablist", "aria-label": "Cities" });
      if (locations().length) {
        locations().forEach((place, idx) => {
          const tab = h("button.note-tab", {
            type: "button",
            role: "tab",
            "aria-selected": String(idx === store.state.weather.active),
            onclick: () => {
              store.update((state) => { state.weather.active = idx; }, "weather");
              draw();
            },
          }, place.name);
          if (store.state.editMode && locations().length > 0) {
            tab.append(h("span.weather-city-x", {
              title: `Remove ${place.name}`,
              onclick: (ev) => {
                ev.stopPropagation();
                store.update((state) => {
                  state.weather.locations.splice(idx, 1);
                  state.weather.active = Math.max(0, Math.min(
                    state.weather.active, state.weather.locations.length - 1));
                }, "weather");
                draw();
              },
            }, " ✕"));
          }
          cityTabs.append(tab);
        });
        cityTabs.append(h("button.note-tab.note-new", {
          type: "button", "aria-label": "Add city",
          onclick: () => { showingSearch = !showingSearch; draw(); },
        }, "+"));
      }

      const changeBtn = h("button.link-btn", {
        type: "button",
        onclick: () => { showingSearch = !showingSearch; draw(); },
      }, showingSearch ? "close" : (locations().length ? "" : "set city"));

      const aqi = data.current.aqi;
      const aqiChip = aqi == null ? null : h("span.aqi-chip", {
        class: `aqi-chip ${aqi <= 50 ? "level-stable" : aqi <= 100 ? "level-watch"
          : aqi <= 150 ? "level-elevated" : "level-critical"} level-chip`,
        title: "US Air Quality Index",
      }, `AQI ${aqi} ${aqi <= 50 ? "GOOD" : aqi <= 100 ? "MODERATE" : aqi <= 150 ? "SENSITIVE" : "UNHEALTHY"}`);

      const head = h("div.weather-now", {},
        h("div.weather-icon", { "aria-hidden": "true" }, icon),
        h("div", {},
          h("div.weather-temp", {}, `${Math.round(data.current.temp)}°`),
          h("div.weather-cond", {}, label),
        ),
        h("div.weather-meta", {},
          h("div.weather-loc", {}, data.location.name, " ", changeBtn),
          h("div.muted", {},
            `Feels ${Math.round(data.current.feels)}° · ${data.current.humidity}% humidity · wind ${Math.round(data.current.wind)} ${data.units.wind}`),
          h("div.weather-extras", {},
            data.sun ? h("span.muted.small", {}, `☀ ${data.sun.sunrise} · ☾ ${data.sun.sunset}`) : null,
            aqiChip,
          ),
        ),
      );

      const daily = h("div.weather-days", {},
        data.daily.slice(0, 7).map((day) => {
          const info = weatherInfo(day.code);
          const weekday = new Date(day.date + "T12:00:00").toLocaleDateString(undefined, { weekday: "short" });
          return h("div.weather-day", { title: `${info.label}, ${day.precipProb ?? 0}% precipitation` },
            h("div.muted.small", {}, weekday),
            h("div", { "aria-hidden": "true" }, info.icon),
            h("div.small", {},
              h("span", {}, `${Math.round(day.max)}°`),
              h("span.muted", {}, ` ${Math.round(day.min)}°`)),
          );
        }),
      );

      clear(body);
      if (locations().length) body.append(cityTabs);
      body.append(head);
      if (showingSearch) {
        body.append(citySearch(ctx, (place) => {
          store.update((state) => {
            const exists = state.weather.locations.findIndex(
              (l) => l.name === place.name && l.lat === place.lat);
            if (exists >= 0) {
              state.weather.active = exists;
            } else {
              state.weather.locations.push(place);
              state.weather.active = state.weather.locations.length - 1;
            }
            state.weather.locations = state.weather.locations.slice(0, 5);
          }, "weather");
          showingSearch = false;
          draw();
        }));
      }
      body.append(tempChart(data.hourly), daily);
    };

    let lastWeather = null;
    ctx.onSummarize(() => lastWeather && ({
      kind: "weather forecast",
      title: lastWeather.location.name,
      content:
        `Now: ${Math.round(lastWeather.current.temp)}°, feels ${Math.round(lastWeather.current.feels)}°, ` +
        `humidity ${lastWeather.current.humidity}%, wind ${Math.round(lastWeather.current.wind)} ${lastWeather.units.wind}.\n` +
        lastWeather.daily.map((d) =>
          `${d.date}: ${Math.round(d.min)}–${Math.round(d.max)}°, precip ${d.precipProb ?? 0}%`).join("\n"),
    }));
    ctx._track = (data) => { lastWeather = data; };

    ctx.onStore((topic) => { if (topic === "editMode") draw(); });
    ctx.onRefresh(draw);
    draw();
    ctx.every(10 * 60_000, draw);
  },
};
