import 'reflect-metadata';
import { container } from 'tsyringe';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { DatabaseService } from '@/infrastructure/database';
import { RedisService } from '@/infrastructure/redis';
import { MCPManager } from '@/integrations/mcp/mcp-manager';
import { NPMPackageManager } from '@/services/npm-package-manager';
import { ProjectManager } from '@/services/project-manager';
import { MemoryService } from '@/services/memory-service';
import { TelegramHandler } from '@/integrations/messengers/telegram-handler';
import { DiscordHandler } from '@/integrations/messengers/discord-handler';
import { SlackHandler } from '@/integrations/messengers/slack-handler';
import { WhatsAppHandler } from '@/integrations/messengers/whatsapp-handler';
import { AgentService } from '@/services/agent-service';

export async function setupDependencies(): Promise<void> {
  const logger = new Logger();
  const config = new ConfigService();

  container.registerInstance(Logger, logger);
  container.registerInstance(ConfigService, config);

  logger.info('Initializing core services...');

  const database = new DatabaseService(config, logger);
  await database.initialize();
  container.registerInstance(DatabaseService, database);

  const redis = new RedisService(config, logger);
  await redis.connect();
  container.registerInstance(RedisService, redis);

  const mcpManager = new MCPManager(config, logger);
  await mcpManager.initialize();
  container.registerInstance(MCPManager, mcpManager);

  const npmManager = new NPMPackageManager(config, logger);
  container.registerInstance(NPMPackageManager, npmManager);

  const projectManager = new ProjectManager(database, logger);
  container.registerInstance(ProjectManager, projectManager);

  const memoryService = new MemoryService(database, logger);
  container.registerInstance(MemoryService, memoryService);

  const agentService = new AgentService(
    mcpManager,
    npmManager,
    projectManager,
    memoryService,
    config,
    logger
  );
  container.registerInstance(AgentService, agentService);

  container.registerSingleton(TelegramHandler);
  container.registerSingleton(DiscordHandler);
  container.registerSingleton(SlackHandler);
  container.registerSingleton(WhatsAppHandler);

  logger.info('Dependency injection configured successfully');
}

export { container };
