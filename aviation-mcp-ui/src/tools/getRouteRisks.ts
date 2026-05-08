import { alertsAlongRoute, findAirport, getRoute } from "../data.js";
import { buildResource, loadTemplate, renderTemplate, type UIResource } from "../render.js";
import { haversineKm } from "../geo.js";
import type { Alert, Airport } from "../types.js";

export interface GetRouteRisksArgs {
  origin: string;
  dest: string;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | UIResource>;
  structuredContent?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

function displayName(a: Airport): string {
  if (a.icao === "KJFK") return "New York JFK";
  return a.name;
}

function severityRank(a: Alert): number {
  return { critical: 0, warning: 1, notice: 2, advisory: 3 }[a.severity];
}

function safeJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

export async function getRouteRisks(args: GetRouteRisksArgs): Promise<ToolResult> {
  const origin = findAirport(args.origin);
  const dest = findAirport(args.dest);
  if (!origin || !dest) {
    throw new Error(`Unknown airport: ${origin ? args.dest : args.origin}. Use 4-letter ICAO codes like KPHX, KJFK, LSZH, or OMDB.`);
  }

  const route = getRoute(origin.icao, dest.icao);
  const alerts = alertsAlongRoute(route.waypoints, 250).sort((a, b) => severityRank(a) - severityRank(b));
  const distanceKm = Math.round(haversineKm([origin.lng, origin.lat], [dest.lng, dest.lat]));
  const html = renderTemplate(loadTemplate("route-map.html"), {
    theme_css: loadTemplate("theme.css"),
    origin: origin.icao,
    dest: dest.icao,
    origin_name: displayName(origin),
    dest_name: displayName(dest),
    distance_km: distanceKm.toLocaleString(),
    alert_count: alerts.length,
    route_json: safeJson(route.waypoints.map(([lng, lat]) => [lat, lng])),
    origin_json: safeJson([origin.lat, origin.lng]),
    dest_json: safeJson([dest.lat, dest.lng]),
    alerts_json: safeJson(
      alerts.map((a) => ({
        id: a.id,
        severity: a.severity,
        headline: a.headline,
        region: a.region,
        tags: a.tags,
        lat: a.lat,
        lng: a.lng,
      }))
    ),
    alerts: alerts.map((a) => ({
      severity: a.severity.toUpperCase(),
      severity_class: `sev-${a.severity}`,
      headline: a.headline,
      region: a.region,
      tags: a.tags.join(" · "),
    })),
  });

  return {
    content: [
      {
        type: "text",
        text: `${origin.icao} to ${dest.icao}: ${alerts.length} alert(s) within the 250 km route corridor.`,
      },
    ],
    structuredContent: {
      view: "route",
      id: `${origin.icao}-${dest.icao}`,
      origin: origin.icao,
      dest: dest.icao,
      alertCount: alerts.length,
    },
    _meta: { html, height: 760 },
  };
}
