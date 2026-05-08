#!/usr/bin/env node
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { registerAppResource, registerAppTool, RESOURCE_MIME_TYPE } from "@modelcontextprotocol/ext-apps/server";
import { createUIResource } from "@mcp-ui/server";
import { z } from "zod";
import { getAirport } from "./tools/getAirport.js";
import { getFir } from "./tools/getFir.js";
import { getFleetStatus } from "./tools/getFleetStatus.js";
import { getRouteRisks } from "./tools/getRouteRisks.js";
import { loadTemplate } from "./render.js";

type ToolHandler<T> = (args: T) => Promise<CallToolResult>;
const WIDGET_URI = "ui://aviation/widget.html";
const toolMeta = {
  securitySchemes: [{ type: "noauth" }],
  ui: { resourceUri: WIDGET_URI },
  "openai/outputTemplate": WIDGET_URI,
  "openai/widgetAccessible": true,
  "openai/toolInvocation/invoking": "Building dashboard",
  "openai/toolInvocation/invoked": "Dashboard ready",
} as const;
const dashboardOutputSchema = {
  view: z.string(),
  id: z.string(),
  title: z.string().optional(),
  summary: z.string().optional(),
  origin: z.string().optional(),
  dest: z.string().optional(),
  alertCount: z.number().optional(),
  total: z.number().optional(),
  issueCount: z.number().optional(),
};

export const toolDefinitions = [
  {
    name: "get_airport",
    title: "Airport Security Deep Dive",
    description: "Render a branded MCP-UI airport risk card for a 4-letter ICAO code such as LSZH.",
    inputSchema: { icao: z.string().min(4).max(4).describe("4-letter ICAO airport code") },
    outputSchema: dashboardOutputSchema,
    _meta: toolMeta,
    handler: getAirport as ToolHandler<{ icao: string }>,
  },
  {
    name: "get_fir",
    title: "FIR Threat Detail",
    description: "Render a branded MCP-UI conflict-zone/FIR threat panel such as ORBB Baghdad FIR.",
    inputSchema: { fir_id: z.string().min(4).max(4).describe("FIR identifier, for example ORBB") },
    outputSchema: dashboardOutputSchema,
    _meta: toolMeta,
    handler: getFir as ToolHandler<{ fir_id: string }>,
  },
  {
    name: "get_fleet_status",
    title: "Fleet Status Board",
    description: "Render a branded MCP-UI fleet board showing aircraft status and active issues.",
    outputSchema: dashboardOutputSchema,
    _meta: toolMeta,
    handler: getFleetStatus as () => Promise<CallToolResult>,
  },
  {
    name: "get_route_risks",
    title: "Route Risk Overlay",
    description: "Render a branded MCP-UI map showing security and aviation alerts along a flight route.",
    inputSchema: {
      origin: z.string().min(4).max(4).describe("Origin ICAO airport code, for example KPHX"),
      dest: z.string().min(4).max(4).describe("Destination ICAO airport code, for example KJFK"),
    },
    outputSchema: dashboardOutputSchema,
    _meta: toolMeta,
    handler: getRouteRisks as ToolHandler<{ origin: string; dest: string }>,
  },
] as const;

export function createAviationServer(): McpServer {
  const server = new McpServer({
    name: "aviation-mcp-ui",
    version: "0.1.0",
  });

  const widgetResource = createUIResource({
    uri: WIDGET_URI,
    content: { type: "rawHtml", htmlString: loadTemplate("chatgpt-widget.html") },
    encoding: "text",
  });

  registerAppResource(
    server,
    "aviation-widget",
    WIDGET_URI,
    {
      title: "Aviation Risk Dashboard",
      mimeType: RESOURCE_MIME_TYPE,
    },
    async () => ({
      contents: [
        {
          ...widgetResource.resource,
          _meta: {
            ui: {
              prefersBorder: true,
              csp: {
                connectDomains: [],
                resourceDomains: [
                  "https://unpkg.com",
                  "https://a.basemaps.cartocdn.com",
                  "https://b.basemaps.cartocdn.com",
                  "https://c.basemaps.cartocdn.com",
                  "https://d.basemaps.cartocdn.com",
                ],
              },
            },
            "openai/widgetDescription": "A branded aviation security dashboard rendered from the latest tool result.",
            "openai/widgetPrefersBorder": true,
            "openai/widgetCSP": {
              connect_domains: [],
              resource_domains: [
                "https://unpkg.com",
                "https://a.basemaps.cartocdn.com",
                "https://b.basemaps.cartocdn.com",
                "https://c.basemaps.cartocdn.com",
                "https://d.basemaps.cartocdn.com",
              ],
            },
          },
        },
      ],
    })
  );

  for (const tool of toolDefinitions) {
    if ("inputSchema" in tool) {
      registerAppTool(
        server,
        tool.name,
        {
          title: tool.title,
          description: tool.description,
          inputSchema: tool.inputSchema,
          outputSchema: tool.outputSchema,
          _meta: tool._meta,
        },
        (args: unknown) => tool.handler(args as never)
      );
    } else {
      registerAppTool(
        server,
        tool.name,
        {
          title: tool.title,
          description: tool.description,
          outputSchema: tool.outputSchema,
          _meta: tool._meta,
        },
        () => tool.handler()
      );
    }
  }

  return server;
}

function setCorsHeaders(res: ServerResponse): void {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, MCP-Protocol-Version, Mcp-Session-Id");
  res.setHeader("Access-Control-Expose-Headers", "Mcp-Session-Id");
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) return undefined;
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function isInitializeBody(body: unknown): boolean {
  const messages = Array.isArray(body) ? body : [body];
  return messages.some(
    (message) =>
      message &&
      typeof message === "object" &&
      "method" in message &&
      (message as { method?: unknown }).method === "initialize"
  );
}

export function createHttpServer() {
  const transports = new Map<string, StreamableHTTPServerTransport>();

  async function createTransport(): Promise<StreamableHTTPServerTransport> {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (sessionId) => {
        transports.set(sessionId, transport);
      },
      onsessionclosed: (sessionId) => {
        if (sessionId) transports.delete(sessionId);
      },
    });
    await createAviationServer().connect(transport);
    return transport;
  }

  return createServer(async (req: IncomingMessage, res: ServerResponse) => {
    setCorsHeaders(res);

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, name: "aviation-mcp-ui" }));
      return;
    }

    if (req.url !== "/mcp") {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "not_found", message: "Use /mcp for MCP requests." }));
      return;
    }

    try {
      const sessionId = req.headers["mcp-session-id"];
      const existingSessionId = Array.isArray(sessionId) ? sessionId[0] : sessionId;
      let transport = existingSessionId ? transports.get(existingSessionId) : undefined;
      let parsedBody: unknown;

      if (req.method === "POST") {
        parsedBody = await readJsonBody(req);
        if (!transport && isInitializeBody(parsedBody)) {
          transport = await createTransport();
        }
      }

      if (!transport) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "missing_session", message: "Initialize a new MCP session with POST /mcp first." }));
        return;
      }

      await transport.handleRequest(req, res, parsedBody);
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: "server_error", message: err instanceof Error ? err.message : String(err) }));
    }
  });
}

export async function mainHttp(): Promise<void> {
  const port = Number(process.env.PORT ?? 2091);
  const host = process.env.HOST ?? "127.0.0.1";
  const server = createHttpServer();
  await new Promise<void>((resolve) => server.listen(port, host, resolve));
  console.error(`aviation-mcp-ui HTTP server listening on http://${host}:${port}/mcp`);
}

export async function main(): Promise<void> {
  const server = createAviationServer();
  await server.connect(new StdioServerTransport());
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const entry = process.argv.includes("--http") ? mainHttp : main;
  entry().catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
}
