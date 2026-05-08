import { alertsLinkedToFir, findFir } from "../data.js";
import { buildResource, loadTemplate, renderTemplate, type UIResource } from "../render.js";
import type { FIR, Weapon } from "../types.js";

export interface GetFirArgs {
  fir_id: string;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | UIResource>;
  structuredContent?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

const WEAPONS: Array<{ id: Weapon; label: string }> = [
  { id: "small_arms", label: "Small Arms" },
  { id: "aaa_light", label: "AAA (Light)" },
  { id: "aaa", label: "AAA" },
  { id: "manpads", label: "MANPADS" },
  { id: "manpads_advanced", label: "MANPADS (Advanced)" },
  { id: "rpg", label: "RPG" },
  { id: "atgm", label: "ATGM" },
  { id: "sam", label: "SAM" },
  { id: "sam_mobile", label: "SAM (Mobile)" },
  { id: "sam_advanced", label: "SAM (Advanced)" },
];

function weaponCells(fir: FIR): Array<{ label: string; class_name: string }> {
  return WEAPONS.map((w) => ({
    label: w.label,
    class_name: fir.weapons.includes(w.id) ? "active" : "inactive",
  }));
}

function polygonCenter(fir: FIR): [number, number] {
  const points = fir.polygon.slice(0, -1);
  const sums = points.reduce(
    (acc, [lng, lat]) => ({ lng: acc.lng + lng, lat: acc.lat + lat }),
    { lng: 0, lat: 0 }
  );
  return [sums.lat / points.length, sums.lng / points.length];
}

function severityClass(fir: FIR): string {
  if (fir.weapons.includes("sam_advanced") || fir.weapons.includes("sam")) return "critical";
  if (fir.hostile_intercepts || fir.weapons.includes("manpads_advanced")) return "warning";
  return "notice";
}

function recommendedPosture(fir: FIR): string {
  if (fir.weapons.includes("sam_advanced") || fir.weapons.includes("sam")) {
    return "Avoid overflight unless mission critical. Route around active missile engagement zones.";
  }
  if (fir.hostile_intercepts || fir.weapons.includes("manpads_advanced")) {
    return `Avoid below FL-${fir.weaponry_range_floor}. Dispatch should review alternate tracks and state advisories.`;
  }
  return "Monitor advisories and maintain standard company routing procedures.";
}

function safeJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

export async function getFir(args: GetFirArgs): Promise<ToolResult> {
  const firId = args.fir_id.toUpperCase();
  const fir = findFir(firId);
  if (!fir) {
    throw new Error(`Unknown FIR: ${args.fir_id}. Try ORBB (Baghdad), OIIX (Tehran), or LSAS (Switzerland).`);
  }

  const alerts = alertsLinkedToFir(fir.id);
  const top = alerts[0];
  const severity = severityClass(fir);
  const center = polygonCenter(fir);
  const html = renderTemplate(loadTemplate("fir-detail.html"), {
    theme_css: loadTemplate("theme.css"),
    id: fir.id,
    name: fir.name,
    country: fir.country,
    issued_by: fir.issued_by,
    date_of_publish: top ? new Date(top.active_from).toUTCString().slice(5, 16).toUpperCase() : "NOT PUBLISHED",
    alert_headline: top?.headline ?? "NO ACTIVE FIR ALERTS",
    alert_time: top ? new Date(top.active_from).toUTCString().slice(5, 22) : "",
    alert_region: top?.region ?? "AIRSPACE",
    alert_count: alerts.length,
    weaponry_range_floor: fir.weaponry_range_floor,
    flight_level_floor: fir.flight_level_floor,
    flight_level_ceiling: fir.flight_level_ceiling,
    hostile_intercepts: fir.hostile_intercepts ? "Hostile Aircraft Intercepts" : "No hostile intercepts published",
    warnings: fir.cz_warnings.length ? fir.cz_warnings : ["No CZ warnings published"],
    weapons: weaponCells(fir),
    severity,
    severity_label: severity.toUpperCase(),
    risk_score: severity === "critical" ? 92 : severity === "warning" ? 76 : 34,
    posture: recommendedPosture(fir),
    confidence: top ? "HIGH CONFIDENCE" : "MEDIUM CONFIDENCE",
    fir_polygon_json: safeJson(fir.polygon.map(([lng, lat]) => [lat, lng])),
    fir_center_json: safeJson(center),
    alert_markers_json: safeJson(alerts.map((a) => ({ lat: a.lat, lng: a.lng, headline: a.headline, severity: a.severity }))),
    alert_items: alerts.map((a) => ({
      severity: a.severity.toUpperCase(),
      headline: a.headline,
      region: a.region,
      active_to: new Date(a.active_to).toUTCString().slice(5, 22),
    })),
  });

  const summary =
    `${fir.id} — ${fir.name}, ${fir.country}. ` +
    `Risk band FL-${fir.flight_level_floor} to FL-${fir.flight_level_ceiling}; ` +
    `${alerts.length} linked aviation alert(s).`;

  return {
    content: [{ type: "text", text: summary }],
    structuredContent: {
      view: "fir",
      id: fir.id,
      title: `${fir.id} ${fir.name}`,
      summary,
      riskScore: severity === "critical" ? 92 : severity === "warning" ? 76 : 34,
    },
    _meta: { html, height: 1120 },
  };
}
