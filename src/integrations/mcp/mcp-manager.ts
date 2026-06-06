import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { ApiClient } from '@/infrastructure/api-client';

export interface MCPServer {
  endpoint: string;
  tools: MCPTool[];
  resources: MCPResource[];
  capabilities: string[];
}

export interface MCPTool {
  name: string;
  description: string;
  parameters?: Record<string, any>;
  returnType?: string;
}

export interface MCPResource {
  name: string;
  description: string;
  type: string;
  uri?: string;
}

export interface MCPRequest {
  jsonrpc: string;
  method: string;
  params: any;
  id: string | number;
}

export interface MCPResponse {
  jsonrpc: string;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string | number;
}

export class MCPManager {
  private servers: Map<string, MCPServer> = new Map();
  private apiClients: Map<string, ApiClient> = new Map();
  private readonly logger: Logger;
  private readonly config: ConfigService;

  constructor(config: ConfigService, logger: Logger) {
    this.config = config;
    this.logger = logger;
  }

  async initialize(): Promise<void> {
    const endpoints = this.config.mcpEndpoints;

    if (!endpoints || endpoints.length === 0) {
      this.logger.warn('No MCP endpoints configured');
      return;
    }

    for (const endpoint of endpoints) {
      await this.connectToEndpoint(endpoint);
    }

    this.logger.info(`Initialized ${this.servers.size} MCP servers`);
  }

  async connectToEndpoint(endpoint: string): Promise<void> {
    try {
      const client = new ApiClient(
        {
          baseURL: endpoint,
          timeout: 10000,
          headers: this.config.mcpApiKey
            ? { Authorization: `Bearer ${this.config.mcpApiKey}` }
            : {},
        },
        this.logger
      );

      this.apiClients.set(endpoint, client);

      const discovery = await this.discoverTools(endpoint, client);
      this.servers.set(endpoint, discovery);

      this.logger.info(`Connected to MCP server: ${endpoint}`, {
        toolCount: discovery.tools.length,
        resourceCount: discovery.resources.length,
      });
    } catch (error) {
      this.logger.error(`Failed to connect to MCP endpoint: ${endpoint}`, error as Error);
    }
  }

  private async discoverTools(endpoint: string, client: ApiClient): Promise<MCPServer> {
    try {
      const tools = await client.post<MCPTool[]>('/discover/tools', {});
      const resources = await client.post<MCPResource[]>('/discover/resources', {});

      return {
        endpoint,
        tools,
        resources,
        capabilities: this.extractCapabilities(tools),
      };
    } catch (error) {
      this.logger.error(`Failed to discover MCP tools from ${endpoint}`, error as Error);
      return {
        endpoint,
        tools: [],
        resources: [],
        capabilities: [],
      };
    }
  }

  private extractCapabilities(tools: MCPTool[]): string[] {
    return tools.map((tool) => tool.name);
  }

  async callTool(
    toolName: string,
    params: Record<string, any>,
    userId: string
  ): Promise<any> {
    for (const [endpoint, server] of this.servers) {
      const tool = server.tools.find((t) => t.name === toolName);
      if (!tool) continue;

      return this.executeToolCall(endpoint, toolName, params, userId);
    }

    throw new Error(`MCP tool not found: ${toolName}`);
  }

  private async executeToolCall(
    endpoint: string,
    toolName: string,
    params: Record<string, any>,
    userId: string
  ): Promise<any> {
    const client = this.apiClients.get(endpoint);
    if (!client) {
      throw new Error(`No client for endpoint: ${endpoint}`);
    }

    const request: MCPRequest = {
      jsonrpc: '2.0',
      method: toolName,
      params,
      id: `${userId}-${Date.now()}-${Math.random()}`,
    };

    try {
      const response = await client.post<MCPResponse>('/execute', request);

      if (response.error) {
        throw new Error(`MCP Error: ${response.error.message}`);
      }

      this.logger.info(`MCP tool executed: ${toolName}`, {
        userId,
        endpoint,
        success: !response.error,
      });

      return response.result;
    } catch (error) {
      this.logger.error(`MCP tool execution failed: ${toolName}`, error as Error);
      throw error;
    }
  }

  getAvailableTools(): MCPTool[] {
    const tools: MCPTool[] = [];
    for (const server of this.servers.values()) {
      tools.push(...server.tools);
    }
    return tools;
  }

  getServers(): MCPServer[] {
    return Array.from(this.servers.values());
  }

  async health(): Promise<boolean> {
    for (const client of this.apiClients.values()) {
      try {
        await client.get('/health');
        return true;
      } catch {
        continue;
      }
    }
    return false;
  }
}
