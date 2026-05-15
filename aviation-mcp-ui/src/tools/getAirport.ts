import { findAirport, alertsNearAirport } from "../data.js";
import { loadTemplate, renderTemplate, buildResource, type UIResource } from "../render.js";
import { haversineKm } from "../geo.js";
import type { Airport } from "../types.js";

export interface GetAirportArgs {
  icao: string;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | UIResource>;
  structuredContent?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

function ratingDeg(level: number): number {
  return Math.max(0, Math.min(5, level)) * 72;
}

function runwaySummary(a: Airport): string {
  const r = a.runways[0];
  if (!r) return "Not published";
  return `RW${r.id} · ${r.length_ft.toLocaleString()} ft × ${r.width_ft} ft @ ${r.elevation_ft.toLocaleString()} ft`;
}

function titleCase(s: string): string {
  return s
    .toLowerCase()
    .replace(/\b([a-z])/g, (_, c) => c.toUpperCase());
}

function groundHandlingSummary(a: Airport): string {
  const parts: string[] = [];
  parts.push(a.ground_handling.slots_required ? "Slots Required" : "No Slots Required");
  parts.push(a.ground_handling.handling_required ? "Handling Required" : "Handling Not Required");
  return parts.join(" · ");
}

export async function getAirport(args: GetAirportArgs): Promise<ToolResult> {
  const airport = findAirport(args.icao);
  if (!airport) {
    throw new Error(`Unknown airport: ${args.icao}. Try a 4-letter ICAO like LSZH (Zurich) or KJFK (New York).`);
  }
  const alerts = alertsNearAirport(airport, 50);
  const top = alerts[0];
  const themeCss = loadTemplate("theme.css");
  const tpl = loadTemplate("airport-card.html");

  const data: Record<string, unknown> = {
    theme_css: themeCss,
    icao: airport.icao,
    iata: airport.iata,
    name: airport.name,
    country: airport.country,
    rating: airport.rating,
    rating_label: airport.rating_label,
    rating_deg: ratingDeg(airport.rating),
    covid_inbound: airport.covid_inbound,
    covid_domestic: airport.covid_domestic,
    covid_deg: ratingDeg(Math.max(airport.covid_inbound, airport.covid_domestic)),
    city_risk_label: airport.city_risk_label,
    city_risk_deg: ratingDeg(airport.city_risk),
    medical_risk_label: airport.medical_risk_label,
    medical_risk_deg: ratingDeg(airport.medical_risk),
    date_of_publish: airport.date_of_publish,
    runway_summary: runwaySummary(airport),
    ground_handling_summary: groundHandlingSummary(airport),
    alert: top
      ? {
          alert_time: new Date(top.active_from).toUTCString().slice(5, 22),
          alert_text: titleCase(top.headline),
          alert_distance_km: Math.round(
            haversineKm([airport.lng, airport.lat], [top.lng, top.lat])
          ),
        }
      : null,
  };

  const html = renderTemplate(tpl, data);
  const summary =
    `${airport.icao} (${airport.iata}) — ${airport.name}, ${airport.country}. ` +
    `Rating ${airport.rating} (${airport.rating_label}). ` +
    (alerts.length > 0 ? `${alerts.length} active alert(s) within 50 km.` : "No active alerts nearby.");

  return {
    content: [{ type: "text", text: summary }],
    structuredContent: {
      view: "airport",
      id: airport.icao,
      title: `${airport.icao} (${airport.iata})`,
      summary,
    },
    _meta: { html, height: 780 },
  };
}
