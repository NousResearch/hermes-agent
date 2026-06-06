import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { MCPManager } from '@/integrations/mcp/mcp-manager';
import { NPMPackageManager } from './npm-package-manager';
import { ProjectManager } from './project-manager';
import { MemoryService } from './memory-service';

export interface AgentContext {
  userId: string;
  platform: string;
  conversationId: number;
  userInput: string;
  projectId?: string;
}

export interface AgentResponse {
  text: string;
  actions: AgentAction[];
  metadata: Record<string, any>;
}

export interface AgentAction {
  type:
    | 'npm_search'
    | 'npm_install'
    | 'project_register'
    | 'mcp_call'
    | 'memory_save'
    | 'memory_retrieve';
  params: Record<string, any>;
  result?: any;
  error?: string;
}

export class AgentService {
  private readonly mcpManager: MCPManager;
  private readonly npmManager: NPMPackageManager;
  private readonly projectManager: ProjectManager;
  private readonly memoryService: MemoryService;
  private readonly config: ConfigService;
  private readonly logger: Logger;

  constructor(
    mcpManager: MCPManager,
    npmManager: NPMPackageManager,
    projectManager: ProjectManager,
    memoryService: MemoryService,
    config: ConfigService,
    logger: Logger
  ) {
    this.mcpManager = mcpManager;
    this.npmManager = npmManager;
    this.projectManager = projectManager;
    this.memoryService = memoryService;
    this.config = config;
    this.logger = logger;
  }

  async processUserMessage(context: AgentContext): Promise<AgentResponse> {
    const actions: AgentAction[] = [];

    try {
      this.logger.info(`Processing message from ${context.userId}`, {
        platform: context.platform,
        projectId: context.projectId,
      });

      const userMemory = await this.memoryService.retrieveAll(context.userId);
      const intent = this.parseIntent(context.userInput);

      this.logger.debug(`Detected intent: ${intent.type}`);

      if (intent.type === 'npm_search') {
        const searchAction = await this.handleNpmSearch(
          intent.query,
          context.userId,
          actions
        );
        return {
          text: this.formatSearchResults(searchAction.result),
          actions,
          metadata: { intent: intent.type },
        };
      }

      if (intent.type === 'project_register') {
        const registerAction = await this.handleProjectRegister(
          context.userId,
          intent.projectPath,
          intent.projectName,
          actions
        );
        return {
          text: registerAction.result.message,
          actions,
          metadata: { intent: intent.type, projectId: registerAction.result.projectId },
        };
      }

      if (intent.type === 'npm_install') {
        if (!context.projectId) {
          return {
            text: '⚠️ No project selected. Use "register project /path/to/project" first.',
            actions,
            metadata: { intent: intent.type },
          };
        }

        const installAction = await this.handleNpmInstall(
          context.projectId,
          intent.packageName,
          intent.version,
          context.userId,
          context.conversationId,
          actions
        );
        return {
          text: installAction.result.message,
          actions,
          metadata: { intent: intent.type },
        };
      }

      if (intent.type === 'mcp_call') {
        const mcpAction = await this.handleMcpCall(
          intent.toolName,
          intent.params,
          context.userId,
          actions
        );
        return {
          text: `MCP Tool Result: ${JSON.stringify(mcpAction.result, null, 2)}`,
          actions,
          metadata: { intent: intent.type },
        };
      }

      const response = await this.generateContextualResponse(
        context.userInput,
        userMemory,
        context.userId
      );

      return {
        text: response,
        actions,
        metadata: { intent: 'general_query' },
      };
    } catch (error) {
      this.logger.error('Agent processing failed', error as Error);
      return {
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        actions,
        metadata: { error: (error as Error).message },
      };
    }
  }

  private parseIntent(
    userInput: string
  ): {
    type: string;
    query?: string;
    packageName?: string;
    version?: string;
    projectPath?: string;
    projectName?: string;
    toolName?: string;
    params?: Record<string, any>;
  } {
    const lowerInput = userInput.toLowerCase();

    if (
      lowerInput.includes('register project') ||
      lowerInput.includes('add project')
    ) {
      const match = userInput.match(
        /(?:register|add)\s+project\s+(?:"([^"]*)"|'([^']*)'|(\S+))(?:\s+as\s+(?:"([^"]*)"|'([^']*)'|(\S+)))?/i
      );
      if (match) {
        const projectPath = match[1] || match[2] || match[3];
        const projectName = match[4] || match[5] || match[6];
        return {
          type: 'project_register',
          projectPath,
          projectName,
        };
      }
    }

    if (
      lowerInput.includes('search') ||
      lowerInput.includes('find') ||
      lowerInput.includes('look for')
    ) {
      const query = userInput.replace(/search|find|look for/gi, '').trim();
      return { type: 'npm_search', query };
    }

    if (
      lowerInput.includes('install') ||
      lowerInput.includes('add') ||
      lowerInput.includes('setup')
    ) {
      const match = userInput.match(/(?:install|add|setup)\s+([^\s]+)(?:@(\S+))?/i);
      if (match) {
        return {
          type: 'npm_install',
          packageName: match[1],
          version: match[2],
        };
      }
    }

    if (lowerInput.includes('call') || lowerInput.includes('execute')) {
      const match = userInput.match(/(?:call|execute)\s+(\w+)/i);
      if (match) {
        return {
          type: 'mcp_call',
          toolName: match[1],
          params: {},
        };
      }
    }

    return { type: 'general' };
  }

  private async handleProjectRegister(
    userId: string,
    projectPath: string,
    projectName: string | undefined,
    actions: AgentAction[]
  ): Promise<AgentAction> {
    try {
      const project = await this.projectManager.registerProject(
        userId,
        projectPath,
        projectName
      );

      const action: AgentAction = {
        type: 'project_register',
        params: { projectPath, projectName },
        result: {
          projectId: project.projectId,
          message: `✅ Project registered: ${project.name}\n\nNow you can install packages with: "install <package-name>"`,
        },
      };

      actions.push(action);

      await this.memoryService.save({
        userId,
        key: `current_project`,
        value: project.projectId,
        ttl: 86400,
      });

      return action;
    } catch (error) {
      const action: AgentAction = {
        type: 'project_register',
        params: { projectPath, projectName },
        error: (error as Error).message,
      };
      actions.push(action);
      throw error;
    }
  }

  private async handleNpmSearch(
    query: string,
    userId: string,
    actions: AgentAction[]
  ): Promise<AgentAction> {
    try {
      const results = await this.npmManager.searchPackages(query, 5);

      const action: AgentAction = {
        type: 'npm_search',
        params: { query },
        result: results,
      };

      actions.push(action);

      await this.memoryService.save({
        userId,
        key: `last_search:${query}`,
        value: results,
        ttl: 3600,
      });

      return action;
    } catch (error) {
      const action: AgentAction = {
        type: 'npm_search',
        params: { query },
        error: (error as Error).message,
      };
      actions.push(action);
      throw error;
    }
  }

  private async handleNpmInstall(
    projectId: string,
    packageName: string,
    version: string | undefined,
    userId: string,
    conversationId: number,
    actions: AgentAction[]
  ): Promise<AgentAction> {
    try {
      const result = await this.projectManager.installPackage(
        projectId,
        packageName,
        version,
        false,
        userId
      );

      const action: AgentAction = {
        type: 'npm_install',
        params: { projectId, packageName, version },
        result,
      };

      actions.push(action);

      await this.memoryService.saveConversationMessage(
        conversationId,
        userId,
        'assistant',
        result.message,
        { action: 'npm_install', success: result.success }
      );

      return action;
    } catch (error) {
      const action: AgentAction = {
        type: 'npm_install',
        params: { projectId, packageName, version },
        error: (error as Error).message,
      };
      actions.push(action);
      throw error;
    }
  }

  private async handleMcpCall(
    toolName: string,
    params: Record<string, any>,
    userId: string,
    actions: AgentAction[]
  ): Promise<AgentAction> {
    try {
      const result = await this.mcpManager.callTool(toolName, params, userId);

      const action: AgentAction = {
        type: 'mcp_call',
        params: { toolName, params },
        result,
      };

      actions.push(action);
      return action;
    } catch (error) {
      const action: AgentAction = {
        type: 'mcp_call',
        params: { toolName, params },
        error: (error as Error).message,
      };
      actions.push(action);
      throw error;
    }
  }

  private async generateContextualResponse(
    userInput: string,
    userMemory: Record<string, any>,
    userId: string
  ): Promise<string> {
    let response = `I understand you're asking: "${userInput}"\n\n`;

    if (Object.keys(userMemory).length > 0) {
      response += `Based on our previous conversations:\n`;
      for (const [key, value] of Object.entries(userMemory)) {
        if (typeof value === 'string' && key !== 'current_project') {
          response += `- ${key}: ${value}\n`;
        }
      }
    }

    response += `\nAvailable commands:\n`;
    response += `• "search <query>" - Search NPM packages\n`;
    response += `• "register project <path>" - Register your project\n`;
    response += `• "install <package>" - Install a package (after registering project)\n`;

    return response;
  }

  private formatSearchResults(results: any): string {
    if (!results.packages || results.packages.length === 0) {
      return 'No packages found. Try a different search query.';
    }

    let output = `Found ${results.total} packages. Here are the top results:\n\n`;

    for (const pkg of results.packages) {
      output += `📦 **${pkg.name}** (v${pkg.version})\n`;
      output += `   ${pkg.description || 'No description'}\n`;
      output += `   Author: ${pkg.author} | License: ${pkg.license}\n\n`;
    }

    return output;
  }
}
