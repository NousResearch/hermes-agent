import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { apiGet, apiPost, ApiError } from "./client";

function mockResponse(
  body: unknown,
  init: { status?: number; contentType?: string } = {},
): Response {
  const status = init.status ?? 200;
  const contentType = init.contentType ?? "application/json";
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(status === 204 ? null : text, {
    status,
    headers: { "content-type": contentType },
  });
}

describe("apiFetch", () => {
  beforeEach(() => {
    (window as unknown as Record<string, unknown>).__HT_SESSION_TOKEN__ = "tok-123";
  });
  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as unknown as Record<string, unknown>).__HT_SESSION_TOKEN__;
  });

  it("injects the session-token header and returns parsed JSON", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(mockResponse({ ok: true, count: 2 }));
    const data = await apiGet<{ ok: boolean; count: number }>("/api/status");
    expect(data).toEqual({ ok: true, count: 2 });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/status");
    expect((init!.headers as Headers).get("X-Hermes-Session-Token")).toBe("tok-123");
    expect(init!.credentials).toBe("include");
  });

  it("falls back to the legacy token global", async () => {
    delete (window as unknown as Record<string, unknown>).__HT_SESSION_TOKEN__;
    (window as unknown as Record<string, unknown>).__HERMES_SESSION_TOKEN__ = "legacy";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse({}));
    await apiGet("/api/status");
    const [, init] = fetchMock.mock.calls[0]!;
    expect((init!.headers as Headers).get("X-Hermes-Session-Token")).toBe("legacy");
    delete (window as unknown as Record<string, unknown>).__HERMES_SESSION_TOKEN__;
  });

  it("serializes a POST body as JSON with the right content-type", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse({ ok: true }));
    await apiPost("/api/model/set", { model: "x" });
    const [, init] = fetchMock.mock.calls[0]!;
    expect(init!.method).toBe("POST");
    expect(init!.body).toBe(JSON.stringify({ model: "x" }));
    expect((init!.headers as Headers).get("Content-Type")).toBe("application/json");
  });

  it("throws ApiError with the server message on non-2xx", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ error: "bad model" }, { status: 400 }),
    );
    await expect(apiGet("/api/model/info")).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      message: "bad model",
    });
  });

  it("returns undefined for 204 No Content", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse(null, { status: 204 }));
    await expect(apiGet("/api/thing")).resolves.toBeUndefined();
  });

  it("redirects to login_url on a gated 401 (does not resolve)", async () => {
    const assign = vi.fn();
    const original = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...original, assign },
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ error: "session_expired", login_url: "/login?next=/x" }, { status: 401 }),
    );
    let settled = false;
    void apiGet("/api/status").then(() => (settled = true));
    // The redirect fires after the response body is parsed (async); flush a
    // few macro/microtask turns so the location.assign has run.
    await new Promise((r) => setTimeout(r, 0));
    expect(assign).toHaveBeenCalledWith("/login?next=/x");
    expect(settled).toBe(false);
    Object.defineProperty(window, "location", { configurable: true, value: original });
  });

  it("surfaces a domain 401 as an ApiError when allowUnauthorized is false", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ error: "no permission" }, { status: 401 }),
    );
    await expect(apiGet("/api/monitor")).rejects.toBeInstanceOf(ApiError);
  });
});
