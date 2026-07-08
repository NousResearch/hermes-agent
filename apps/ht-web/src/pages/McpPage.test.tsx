import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import McpPage from "./McpPage";
import type { McpServer } from "@/api/mcp";

// Mock the api module so the page renders against fixed data without network.
vi.mock("@/api/mcp", () => ({
  getMcpServers: vi.fn(),
  getMcpCatalog: vi.fn(),
  addMcpServer: vi.fn(),
  removeMcpServer: vi.fn(),
  testMcpServer: vi.fn(),
  setMcpServerEnabled: vi.fn(),
  installMcpCatalogEntry: vi.fn(),
}));

import { getMcpServers } from "@/api/mcp";

afterEach(cleanup);

function server(partial: Partial<McpServer> & Pick<McpServer, "name">): McpServer {
  return {
    transport: "http",
    url: null,
    command: null,
    args: [],
    env: {},
    auth: null,
    enabled: true,
    tools: null,
    ...partial,
  };
}

describe("McpPage", () => {
  it("renders a server row with its status", async () => {
    vi.mocked(getMcpServers).mockResolvedValue({
      servers: [server({ name: "weather", transport: "http", enabled: true, tools: ["forecast"] })],
    });

    render(<McpPage />);

    // Name shows once the resource resolves.
    expect(await screen.findByText("weather")).toBeTruthy();
    // Connected status chip reflects the tool count.
    expect(screen.getByText(/connected · 1 tool/)).toBeTruthy();
  });
});
