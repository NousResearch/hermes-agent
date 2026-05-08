import { describe, it, expect } from "vitest";
import { haversineKm } from "../src/geo.js";

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
