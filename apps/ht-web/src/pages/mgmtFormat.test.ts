import { describe, it, expect, vi, afterEach } from "vitest";
import { fmtBytes, fmtNumber, fmtCost, relTime } from "./mgmtFormat";

describe("fmtBytes", () => {
  it("formats byte sizes with scaled units", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(2048)).toBe("2.0 KB");
    expect(fmtBytes(1024 * 1024)).toBe("1.0 MB");
    expect(fmtBytes(5 * 1024 * 1024 * 1024)).toBe("5.0 GB");
  });

  it("renders null/undefined/negative as an em dash", () => {
    expect(fmtBytes(null)).toBe("—");
    expect(fmtBytes(undefined)).toBe("—");
    expect(fmtBytes(-1)).toBe("—");
  });
});

describe("fmtNumber", () => {
  it("keeps small numbers verbatim and compacts large ones", () => {
    expect(fmtNumber(0)).toBe("0");
    expect(fmtNumber(999)).toBe("999");
    expect(fmtNumber(1234)).toBe("1.2K");
    expect(fmtNumber(3_400_000)).toBe("3.4M");
    expect(fmtNumber(2_000_000_000)).toBe("2.0B");
  });

  it("renders nullish as an em dash", () => {
    expect(fmtNumber(null)).toBe("—");
    expect(fmtNumber(undefined)).toBe("—");
  });
});

describe("fmtCost", () => {
  it("uses more precision below a dollar", () => {
    expect(fmtCost(0)).toBe("$0.00");
    expect(fmtCost(0.0123)).toBe("$0.0123");
    expect(fmtCost(12.5)).toBe("$12.50");
  });

  it("renders nullish as an em dash", () => {
    expect(fmtCost(null)).toBe("—");
  });
});

describe("relTime", () => {
  afterEach(() => vi.useRealTimers());

  it("formats relative durations from epoch seconds", () => {
    vi.useFakeTimers();
    const now = 1_000_000; // seconds
    vi.setSystemTime(now * 1000);
    expect(relTime(now - 5)).toBe("5s ago");
    expect(relTime(now - 120)).toBe("2m ago");
    expect(relTime(now - 7200)).toBe("2h ago");
    expect(relTime(now - 3 * 86400)).toBe("3d ago");
  });

  it("renders falsy timestamps as an em dash", () => {
    expect(relTime(0)).toBe("—");
    expect(relTime(null)).toBe("—");
  });
});
