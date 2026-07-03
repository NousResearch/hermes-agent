import { describe, expect, it } from "vitest";
import { approveActionEnabled, freshnessState, gatewayValue } from "./appModel";
import { type CapabilityItem, type StatusSnapshot } from "./api";

function cap(over: Partial<CapabilityItem> = {}): CapabilityItem {
  return { id: "approve-action", label: "x", enabled: true, mode: "owner-confirmed-action", reason: "", ...over };
}

describe("approveActionEnabled", () => {
  it("is true only for an enabled owner-confirmed approve-action capability", () => {
    expect(approveActionEnabled([cap()])).toBe(true);
  });

  it("is false when the capability is disabled, blocked, or missing", () => {
    expect(approveActionEnabled([cap({ enabled: false })])).toBe(false);
    expect(approveActionEnabled([cap({ mode: "blocked" })])).toBe(false);
    expect(approveActionEnabled([cap({ id: "status-read" })])).toBe(false);
    expect(approveActionEnabled([])).toBe(false);
  });
});

describe("freshnessState", () => {
  it("reports mock, refreshing, stale, offline and fresh honestly", () => {
    expect(freshnessState("mock", false, null, 0)).toBe("mock");
    expect(freshnessState("connecting", false, null, 0)).toBe("refreshing");
    expect(freshnessState("connected", true, 1000, 1000)).toBe("refreshing");
    expect(freshnessState("offline", false, null, 0)).toBe("offline");
    expect(freshnessState("offline", false, 1000, 2000)).toBe("stale");
    // Older than STALE_AFTER_MS (45s) → stale even while connected.
    expect(freshnessState("connected", false, 1000, 51_000)).toBe("stale");
    expect(freshnessState("connected", false, 1000, 1000)).toBe("fresh");
  });
});

describe("gatewayValue", () => {
  it("mirrors the real gateway running flag", () => {
    const base = { gateway: { running: true } } as unknown as StatusSnapshot;
    expect(gatewayValue(base)).toBe("Готов");
    expect(gatewayValue({ ...base, gateway: { running: false } } as StatusSnapshot)).toBe("Офлайн");
    expect(gatewayValue(null)).toBe("Готов");
  });
});
