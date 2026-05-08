import { describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createHttpServer, toolDefinitions } from "../src/server.js";
import { getAirport } from "../src/tools/getAirport.js";

describe("ChatGPT app rendering contract", () => {
  it("advertises a UI resource on each tool descriptor", () => {
    for (const tool of toolDefinitions) {
      expect(tool._meta?.ui?.resourceUri).toBe("ui://aviation/widget.html");
      expect(tool._meta?.["openai/outputTemplate"]).toBe("ui://aviation/widget.html");
      expect(tool.outputSchema).toBeDefined();
      expect(tool._meta?.securitySchemes).toEqual([{ type: "noauth" }]);
    }
  });

  it("returns generated dashboard HTML in private tool result metadata", async () => {
    const result = await getAirport({ icao: "LSZH" });
    expect(result.content).toEqual([{ type: "text", text: expect.stringContaining("LSZH") }]);
    expect(result.structuredContent).toMatchObject({ view: "airport", id: "LSZH" });
    expect(result._meta?.html).toContain("<!DOCTYPE html>");
  });

  it("serves the ChatGPT widget resource over HTTP", async () => {
    const httpServer = createHttpServer();
    await new Promise<void>((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
    const address = httpServer.address();
    if (!address || typeof address === "string") throw new Error("expected TCP address");

    const client = new Client({ name: "resource-test", version: "0.0.0" });
    await client.connect(
      new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${address.port}/mcp`))
    );
    const resource = await client.readResource({ uri: "ui://aviation/widget.html" });
    expect(resource.contents[0].mimeType).toBe("text/html;profile=mcp-app");
    expect(resource.contents[0]._meta?.ui).toMatchObject({
      prefersBorder: true,
    });
    expect(resource.contents[0].text).toContain("toolResponseMetadata");
    expect(resource.contents[0].text).toContain("notifyIntrinsicHeight");
    expect(resource.contents[0].text).toContain("dashboard-frame");
    expect(resource.contents[0].text).toContain("openai:set_globals");
    expect(resource.contents[0].text).toContain("waitForDashboardPayload");
    expect(resource.contents[0].text).toContain("readDashboardHeight");
    expect(resource.contents[0].text).toContain("setFrameHeight");

    await client.close();
    await new Promise<void>((resolve, reject) =>
      httpServer.close((err) => (err ? reject(err) : resolve()))
    );
  });
});
