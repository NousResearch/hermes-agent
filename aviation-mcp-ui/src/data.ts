import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { Airport, FIR, Alert, Aircraft, Route } from "./types.js";
import { haversineKm, greatCircleWaypoints, pointNearPolyline } from "./geo.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FIXTURE_DIR = join(__dirname, "..", "fixtures");

function load<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURE_DIR, name), "utf8")) as T;
}

const airports: Airport[] = load("airports.json");
const firs: FIR[] = load("firs.json");
const alerts: Alert[] = load("alerts.json");
const fleet: Aircraft[] = load("fleet.json");
const routes: Route[] = load("routes.json");

export function findAirport(icao: string): Airport | undefined {
  const u = icao.toUpperCase();
  return airports.find((a) => a.icao === u);
}

export function findFir(id: string): FIR | undefined {
  const u = id.toUpperCase();
  return firs.find((f) => f.id === u);
}

export function getFleet(): Aircraft[] {
  return fleet;
}

export function alertsNearAirport(airport: Airport, radiusKm = 50): Alert[] {
  return alerts.filter((al) => {
    if (al.linked_airport_icao === airport.icao) return true;
    return haversineKm([airport.lng, airport.lat], [al.lng, al.lat]) <= radiusKm;
  });
}

export function alertsLinkedToFir(firId: string): Alert[] {
  const u = firId.toUpperCase();
  return alerts.filter((al) => al.linked_fir_id === u);
}

export function getRoute(origin: string, dest: string): Route {
  const o = origin.toUpperCase();
  const d = dest.toUpperCase();
  const cached = routes.find((r) => r.origin === o && r.dest === d);
  if (cached) return cached;
  const a = findAirport(o);
  const b = findAirport(d);
  if (!a || !b) {
    throw new Error(`Unknown airport: ${a ? d : o}`);
  }
  return {
    origin: o,
    dest: d,
    waypoints: greatCircleWaypoints([a.lng, a.lat], [b.lng, b.lat], 12),
  };
}

export function alertsAlongRoute(
  waypoints: [number, number][],
  corridorKm = 250
): Alert[] {
  return alerts.filter((al) =>
    pointNearPolyline([al.lng, al.lat], waypoints, corridorKm)
  );
}
