import { describe, it, expect } from "vitest";
import { getAirport } from "../src/tools/getAirport.js";
import { getFir } from "../src/tools/getFir.js";
import { getFleetStatus } from "../src/tools/getFleetStatus.js";
import { getRouteRisks } from "../src/tools/getRouteRisks.js";

describe("getAirport", () => {
  it("returns text plus ChatGPT component metadata for a known ICAO", async () => {
    const r = await getAirport({ icao: "LSZH" });
    const first = r.content[0];
    if (first.type !== "text") throw new Error("expected text first");
    expect(first.text).toContain("LSZH");
    expect(r.content).toHaveLength(1);
    expect(r.structuredContent).toMatchObject({ view: "airport", id: "LSZH" });
    expect(r._meta?.html).toContain("Zurich");
    expect(r._meta?.html).toContain("12,139");
    expect(r._meta?.height).toBeGreaterThan(600);
  });

  it("includes a linked alert when present", async () => {
    const r = await getAirport({ icao: "LSZH" });
    expect(r._meta?.html).toContain("Hantavirus");
  });

  it("throws a helpful error for unknown ICAO", async () => {
    await expect(getAirport({ icao: "XXXX" })).rejects.toThrow(/Unknown airport/);
  });

  it("escapes HTML in fixture-derived strings", async () => {
    const r = await getAirport({ icao: "LSZH" });
    expect(r._meta?.html).not.toContain("<script>alert");
  });
});

describe("getFir", () => {
  it("returns the Baghdad FIR threat detail resource", async () => {
    const r = await getFir({ fir_id: "ORBB" });
    const first = r.content[0];
    if (first.type !== "text") throw new Error("expected text first");
    expect(first.text).toContain("ORBB");
    expect(r.structuredContent).toMatchObject({ view: "fir", id: "ORBB" });
    expect(r._meta?.html).toContain("Baghdad FIR");
    expect(r._meta?.html).toContain("FL-260");
    expect(r._meta?.html).toContain("AVIATION ALERT");
    expect(r._meta?.html).toContain("fir-map");
    expect(r._meta?.html).toContain("firPolygon");
    expect(r._meta?.html).toContain("Recommended routing posture");
    expect(r._meta?.html).toContain("skeleton");
    expect(r._meta?.height).toBeGreaterThan(900);
  });

  it("throws a helpful error for unknown FIR ids", async () => {
    await expect(getFir({ fir_id: "ZZZZ" })).rejects.toThrow(/Unknown FIR/);
  });
});

describe("getFleetStatus", () => {
  it("returns a fleet grid with issue aircraft highlighted", async () => {
    const r = await getFleetStatus();
    expect(r.structuredContent).toMatchObject({ view: "fleet", id: "status" });
    expect(r._meta?.html).toContain("N787LR");
    expect(r._meta?.html).toContain("DIVERTING");
    expect(r._meta?.html).toContain("pulse-red");
  });
});

describe("getRouteRisks", () => {
  it("returns a route map with alerts along PHX to JFK", async () => {
    const r = await getRouteRisks({ origin: "KPHX", dest: "KJFK" });
    const first = r.content[0];
    if (first.type !== "text") throw new Error("expected text first");
    expect(first.text).toContain("KPHX");
    expect(first.text).toContain("KJFK");
    expect(r.structuredContent).toMatchObject({ view: "route", id: "KPHX-KJFK" });
    expect(r._meta?.html).toContain("Leaflet");
    expect(r._meta?.html).toContain("Phoenix");
    expect(r._meta?.html).toContain("New York JFK");
  });

  it("throws a helpful error for unknown route airports", async () => {
    await expect(getRouteRisks({ origin: "KPHX", dest: "ZZZZ" })).rejects.toThrow(/Unknown airport/);
  });
});
