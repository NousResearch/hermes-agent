import { getFleet } from "../data.js";
import { buildResource, loadTemplate, renderTemplate, type UIResource } from "../render.js";
import type { Aircraft, AircraftStatus, IssueSeverity } from "../types.js";

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | UIResource>;
  structuredContent?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

function statusLabel(status: AircraftStatus): string {
  return status.replace("_", " ").toUpperCase();
}

function dotClass(status: AircraftStatus, severity?: IssueSeverity): string {
  if (status === "issue" || severity === "critical") return "dot-red";
  if (status === "maintenance" || severity === "warning") return "dot-amber";
  if (status === "in_flight") return "dot-green";
  return "dot-grey";
}

function locationLabel(a: Aircraft): string {
  if (a.location.icao) return a.location.icao;
  if (a.location.lat !== undefined && a.location.lng !== undefined) {
    return `${a.location.lat.toFixed(1)}, ${a.location.lng.toFixed(1)}`;
  }
  return "Unknown";
}

export async function getFleetStatus(): Promise<ToolResult> {
  const fleet = getFleet();
  const issueCount = fleet.filter((a) => a.status === "issue" || a.issue?.severity === "critical").length;
  const html = renderTemplate(loadTemplate("fleet-grid.html"), {
    theme_css: loadTemplate("theme.css"),
    total: fleet.length,
    issue_count: issueCount,
    in_flight_count: fleet.filter((a) => a.status === "in_flight").length,
    cards: fleet.map((a) => ({
      tail_number: a.tail_number,
      type: a.type,
      callsign: a.callsign,
      status: statusLabel(a.status),
      status_class: dotClass(a.status, a.issue?.severity),
      card_class: a.status === "issue" ? "issue pulse-red" : "",
      location: locationLabel(a),
      route: a.origin && a.dest ? `${a.origin} → ${a.dest}` : "No active route",
      issue_headline: a.issue?.headline ?? "",
      issue_detail: a.issue?.detail ?? "",
      has_issue: Boolean(a.issue),
    })),
  });

  return {
    content: [
      {
        type: "text",
        text: `Fleet status: ${fleet.length} aircraft monitored, ${issueCount} requiring attention.`,
      },
    ],
    structuredContent: {
      view: "fleet",
      id: "status",
      total: fleet.length,
      issueCount,
    },
    _meta: { html, height: 900 },
  };
}
