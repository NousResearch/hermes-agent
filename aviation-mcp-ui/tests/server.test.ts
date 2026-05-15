import { describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createHttpServer, createAviationServer, toolDefinitions } from "../src/server.js";

describe("MCP server wiring", () => {
  it("registers the four aviation UI tools", () => {
    expect(toolDefinitions.map((t) => t.name)).toEqual([
      "get_airport",
      "get_fir",
      "get_fleet_status",
      "get_route_risks",
    ]);
  });

  it("creates an MCP server without connecting a transport", () => {
    const server = createAviationServer();
    expect(server.isConnected()).toBe(false);
  });

  it("serves tools over streamable HTTP at /mcp", async () => {
    const httpServer = createHttpServer();
    await new Promise<void>((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
    const address = httpServer.address();
    if (!address || typeof address === "string") throw new Error("expected TCP address");

    const client = new Client({ name: "http-test", version: "0.0.0" });
    await client.connect(
      new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${address.port}/mcp`))
    );
    const tools = await client.listTools();
    expect(tools.tools.map((t) => t.name)).toContain("get_route_risks");

    await client.close();
    const secondClient = new Client({ name: "http-test-2", version: "0.0.0" });
    await secondClient.connect(
      new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${address.port}/mcp`))
    );
    const secondTools = await secondClient.listTools();
    expect(secondTools.tools.map((t) => t.name)).toContain("get_airport");
    await secondClient.close();

    await new Promise<void>((resolve, reject) =>
      httpServer.close((err) => (err ? reject(err) : resolve()))
    );
  });
});
