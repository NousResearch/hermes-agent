import 'dotenv/config';
import 'reflect-metadata';
import express, { Express } from 'express';
import { setupDependencies, container } from '@/di-container';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { DatabaseService } from '@/infrastructure/database';
import { RedisService } from '@/infrastructure/redis';
import { TelegramHandler } from '@/integrations/messengers/telegram-handler';
import { DiscordHandler } from '@/integrations/messengers/discord-handler';
import { SlackHandler } from '@/integrations/messengers/slack-handler';
import { WhatsAppHandler } from '@/integrations/messengers/whatsapp-handler';

let app: Express;
let logger: Logger;

async function bootstrap(): Promise<void> {
  try {
    logger = new Logger('HermesAgent');
    logger.info('🚀 Hermes Agent starting...');

    await setupDependencies();

    const config = container.resolve(ConfigService);
    app = express();

    setupMiddleware();
    setupRoutes();

    await startMessengers(config, logger);

    const port = config.port;
    app.listen(port, () => {
      logger.info(`✅ Hermes Agent running on port ${port}`);
    });

    setupGracefulShutdown();
  } catch (error) {
    logger?.error('Bootstrap failed', error as Error);
    process.exit(1);
  }
}

function setupMiddleware(): void {
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  app.use((req, res, next) => {
    logger.debug(`${req.method} ${req.path}`);
    next();
  });
}

function setupRoutes(): void {
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  app.get('/api/status', (req, res) => {
    res.json({
      status: 'running',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
    });
  });

  app.post('/api/message', async (req, res) => {
    try {
      const { userId, platform, message } = req.body;

      if (!userId || !platform || !message) {
        return res.status(400).json({ error: 'Missing required fields' });
      }

      const agentService = container.resolve(AgentService);
      const memoryService = container.resolve(MemoryService);

      const conversationId = await memoryService.createConversation(
        userId,
        platform
      );

      const response = await agentService.processUserMessage({
        userId,
        platform,
        conversationId,
        userInput: message,
      });

      res.json(response);
    } catch (error) {
      logger.error('API message processing failed', error as Error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/npm/search', async (req, res) => {
    try {
      const { q, limit } = req.query;

      if (!q || typeof q !== 'string') {
        return res.status(400).json({ error: 'Query parameter required' });
      }

      const npmManager = container.resolve(NPMPackageManager);
      const results = await npmManager.searchPackages(q, parseInt(limit as string) || 10);

      res.json(results);
    } catch (error) {
      logger.error('NPM search API failed', error as Error);
      res.status(500).json({ error: 'Search failed' });
    }
  });

  app.post('/api/npm/install', async (req, res) => {
    try {
      const { packageName, version, userId } = req.body;

      if (!packageName) {
        return res.status(400).json({ error: 'packageName required' });
      }

      const npmManager = container.resolve(NPMPackageManager);
      const result = await npmManager.installPackage(packageName, version, userId);

      res.json(result);
    } catch (error) {
      logger.error('NPM install API failed', error as Error);
      res.status(500).json({ error: 'Installation failed' });
    }
  });
}

async function startMessengers(config: ConfigService, logger: Logger): Promise<void> {
  const agentService = container.resolve(AgentService);
  const memoryService = container.resolve(MemoryService);

  if (config.telegramToken) {
    try {
      const telegramHandler = new TelegramHandler(
        config,
        logger,
        agentService,
        memoryService
      );
      await telegramHandler.start();
      logger.info('✅ Telegram messenger initialized');
    } catch (error) {
      logger.warn('Telegram initialization skipped', (error as Error).message);
    }
  }

  if (config.discordToken) {
    try {
      const discordHandler = new DiscordHandler(
        config,
        logger,
        agentService,
        memoryService
      );
      await discordHandler.start();
      logger.info('✅ Discord messenger initialized');
    } catch (error) {
      logger.warn('Discord initialization skipped', (error as Error).message);
    }
  }

  if (config.slackBotToken) {
    try {
      const slackHandler = new SlackHandler(config, logger, agentService, memoryService);
      await slackHandler.start(config.port + 1);
      logger.info('✅ Slack messenger initialized');
    } catch (error) {
      logger.warn('Slack initialization skipped', (error as Error).message);
    }
  }
}

function setupGracefulShutdown(): void {
  process.on('SIGINT', async () => {
    logger.info('Shutting down gracefully...');

    try {
      const database = container.resolve(DatabaseService);
      const redis = container.resolve(RedisService);

      await database.close();
      await redis.close();

      logger.info('✅ Services closed');
      process.exit(0);
    } catch (error) {
      logger.error('Shutdown error', error as Error);
      process.exit(1);
    }
  });
}

import { AgentService } from '@/services/agent-service';
import { MemoryService } from '@/services/memory-service';
import { NPMPackageManager } from '@/services/npm-package-manager';

bootstrap();
