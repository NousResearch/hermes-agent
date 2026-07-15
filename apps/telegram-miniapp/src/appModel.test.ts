import { describe, expect, it } from "vitest";
import { actionFooterLabel, actionValue, approveActionEnabled, freshnessState, gatewayValue } from "./appModel";
import { type CapabilityItem, type StatusSnapshot } from "./api";

function cap(over: Partial<CapabilityItem> = {}): CapabilityItem {
  return { id: "approve-action", label: "x", enabled: true, mode: "owner-confirmed-action", reason: "", ...over };
}

function statusSnapshot(miniapp: Partial<StatusSnapshot["miniapp"]>): StatusSnapshot {
  return {
    ok: true,
    updated_at: "now",
    hermes_home: "configured",
    gateway: { running: true, state: "ok", busy: false, drainable: true, active_agents: 0, restart_requested: false },
    miniapp: { mode: "read-only", actions_enabled: false, public_exposure: false, ...miniapp },
  };
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

describe("action state copy", () => {
  it("does not report actions as just on when decisions are record-only", () => {
    const recordOnly = statusSnapshot({ actions_enabled: true, gateway_resolver_active: false, action_application: "record-only" });
    expect(actionValue(recordOnly)).toBe("Запись");
    expect(actionFooterLabel(recordOnly)).toContain("решения записываются");
    expect(actionFooterLabel(recordOnly)).toContain("применение не подключено");
  });

  it("reports live only with an active resolver heartbeat", () => {
    expect(actionValue(statusSnapshot({ actions_enabled: true, gateway_resolver_active: false, action_application: "live" }))).toBe("Блок");
    expect(actionValue(statusSnapshot({ actions_enabled: true, gateway_resolver_active: true, action_application: "live" }))).toBe("Live");
  });
});
