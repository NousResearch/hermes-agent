import { describe, it, expect } from "vitest";
import { haversineKm, greatCircleWaypoints, pointNearPolyline } from "../src/geo.js";

describe("haversineKm", () => {
  it("returns 0 for identical points", () => {
    expect(haversineKm([0, 0], [0, 0])).toBeCloseTo(0, 5);
  });

  it("computes PHX -> JFK ~3447 km within 1%", () => {
    const phx: [number, number] = [-112.0078, 33.4343];
    const jfk: [number, number] = [-73.7781, 40.6413];
    const d = haversineKm(phx, jfk);
    expect(d).toBeGreaterThan(3413);
    expect(d).toBeLessThan(3481);
  });

  it("computes LHR -> JFK ~5550 km within 1%", () => {
    const lhr: [number, number] = [-0.4543, 51.4700];
    const jfk: [number, number] = [-73.7781, 40.6413];
    const d = haversineKm(lhr, jfk);
    expect(d).toBeGreaterThan(5494);
    expect(d).toBeLessThan(5605);
  });
});

describe("greatCircleWaypoints", () => {
  it("returns n points", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 12);
    expect(pts).toHaveLength(12);
  });

  it("first and last points match endpoints (within 0.01 deg)", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 8);
    expect(pts[0][0]).toBeCloseTo(-112, 1);
    expect(pts[0][1]).toBeCloseTo(33, 1);
    expect(pts[pts.length - 1][0]).toBeCloseTo(-74, 1);
    expect(pts[pts.length - 1][1]).toBeCloseTo(41, 1);
  });

  it("interior points lie roughly on the arc (latitude bows slightly north for east-west)", () => {
    const pts = greatCircleWaypoints([-112, 33], [-74, 41], 11);
    const mid = pts[5];
    const directMidLat = (33 + 41) / 2;
    expect(mid[1]).toBeGreaterThan(directMidLat - 1);
    expect(mid[1]).toBeLessThan(directMidLat + 5);
  });
});

describe("pointNearPolyline", () => {
  const phx: [number, number] = [-112.0078, 33.4343];
  const jfk: [number, number] = [-73.7781, 40.6413];
  const route: [number, number][] = [phx, [-90, 38], jfk];

  it("returns true for a point exactly on a vertex", () => {
    expect(pointNearPolyline([-90, 38], route, 50)).toBe(true);
  });

  it("returns true for a point within threshold", () => {
    // ~80km north of midpoint
    expect(pointNearPolyline([-90, 38.7], route, 200)).toBe(true);
  });

  it("returns false for a far-away point", () => {
    // somewhere over Mexico
    expect(pointNearPolyline([-100, 20], route, 200)).toBe(false);
  });
});
