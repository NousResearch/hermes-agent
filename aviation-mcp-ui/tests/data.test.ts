import { describe, it, expect } from "vitest";
import {
  findAirport,
  findFir,
  alertsNearAirport,
  alertsLinkedToFir,
  alertsAlongRoute,
  getRoute,
  getFleet,
} from "../src/data.js";

describe("data lookups", () => {
  it("finds an airport by ICAO", () => {
    const a = findAirport("LSZH");
    expect(a?.iata).toBe("ZRH");
  });

  it("returns undefined for unknown ICAO", () => {
    expect(findAirport("XXXX")).toBeUndefined();
  });

  it("finds a FIR by id", () => {
    expect(findFir("ORBB")?.name).toBe("Baghdad FIR");
  });

  it("returns alerts within 50km of an airport", () => {
    const lszh = findAirport("LSZH")!;
    const a = alertsNearAirport(lszh, 50);
    expect(a.some((x) => x.id === "AL-2026-0506")).toBe(true);
  });

  it("returns alerts linked to a FIR", () => {
    const a = alertsLinkedToFir("ORBB");
    expect(a.some((x) => x.id === "AL-2026-0502")).toBe(true);
  });

  it("uses precomputed route when present", () => {
    const r = getRoute("KPHX", "KJFK");
    expect(r.waypoints[0]).toEqual([-112.0078, 33.4343]);
    expect(r.waypoints.at(-1)).toEqual([-73.7781, 40.6413]);
  });

  it("falls back to great-circle for unknown pairs", () => {
    const r = getRoute("KSEA", "KMIA");
    expect(r.waypoints.length).toBeGreaterThan(2);
  });

  it("alertsAlongRoute returns alerts within corridor", () => {
    const r = getRoute("KPHX", "KJFK");
    const a = alertsAlongRoute(r.waypoints, 250);
    expect(a.length).toBeGreaterThan(0);
  });

  it("returns the fleet", () => {
    const f = getFleet();
    expect(f.length).toBeGreaterThanOrEqual(8);
    expect(f.some((a) => a.status === "issue")).toBe(true);
  });
});
