
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

function createServer() {
    const server = new McpServer({
        name: "cosmos-proxy",
        version: "1.0.0",
    });

    server.tool(
        "cosmos_remember",
        "Store information in memory",
        {
            content: z.string().describe("Information to remember"),
            tags: z.string().describe("Comma-separated tags")
        },
        async ({ content, tags }) => {
            return { content: [{ type: "text", text: "Proxied" }] };
        }
    );

    return server;
}

// Hook for Smithery scanner
export function createSandboxServer() {
    return createServer();
}

async function main() {
    const server = createServer();
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

// Run main
main().catch(console.error);


