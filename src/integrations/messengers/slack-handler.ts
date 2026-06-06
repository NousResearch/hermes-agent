import { App } from '@slack/bolt';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { AgentService } from '@/services/agent-service';
import { MemoryService } from '@/services/memory-service';

export class SlackHandler {
  private app: App;
  private readonly logger: Logger;
  private readonly config: ConfigService;
  private readonly agentService: AgentService;
  private readonly memoryService: MemoryService;

  constructor(
    config: ConfigService,
    logger: Logger,
    agentService: AgentService,
    memoryService: MemoryService
  ) {
    this.config = config;
    this.logger = logger;
    this.agentService = agentService;
    this.memoryService = memoryService;

    if (!config.slackBotToken || !config.slackSigningSecret) {
      throw new Error('SLACK_BOT_TOKEN or SLACK_SIGNING_SECRET not configured');
    }

    this.app = new App({
      token: config.slackBotToken,
      signingSecret: config.slackSigningSecret,
    });
  }

  async start(port: number = 3000): Promise<void> {
    this.setupHandlers();

    try {
      await this.app.start(port);
      this.logger.info(`Slack bot started on port ${port}`);
    } catch (error) {
      this.logger.error('Failed to start Slack bot', error as Error);
      throw error;
    }
  }

  private setupHandlers(): void {
    this.app.command('/search', async ({ command, ack, say }) => {
      await ack();

      const userId = command.user_id;
      const query = command.text;

      if (!query) {
        await say('Please provide a search query. Example: `/search express`');
        return;
      }

      const conversationId = await this.memoryService.createConversation(
        userId,
        'slack',
        `Search: ${query}`
      );

      try {
        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'slack',
          conversationId,
          userInput: `search ${query}`,
        });

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'assistant',
          response.text
        );

        await say(response.text);
      } catch (error) {
        this.logger.error('Slack search failed', error as Error);
        await say('❌ Search failed. Please try again.');
      }
    });

    this.app.command('/install', async ({ command, ack, say }) => {
      await ack();

      const userId = command.user_id;
      const packageName = command.text;

      if (!packageName) {
        await say('Please provide a package name. Example: `/install express`');
        return;
      }

      const conversationId = await this.memoryService.createConversation(
        userId,
        'slack',
        `Install: ${packageName}`
      );

      try {
        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'slack',
          conversationId,
          userInput: `install ${packageName}`,
        });

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'assistant',
          response.text
        );

        await say(response.text);
      } catch (error) {
        this.logger.error('Slack install failed', error as Error);
        await say('❌ Installation failed. Please try again.');
      }
    });

    this.app.message(/.*/, async ({ message, say }) => {
      if (!('text' in message) || message.bot_id) return;

      const userId = message.user;
      const userInput = message.text;

      if (userInput.startsWith('/')) return;

      const conversationId = await this.memoryService.createConversation(
        userId,
        'slack'
      );

      try {
        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'slack',
          conversationId,
          userInput,
        });

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'user',
          userInput
        );

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'assistant',
          response.text
        );

        await say(response.text);
      } catch (error) {
        this.logger.error('Slack message processing failed', error as Error);
        await say('❌ Error processing your message. Please try again.');
      }
    });
  }

  async stop(): Promise<void> {
    this.logger.info('Slack bot stopped');
  }
}
