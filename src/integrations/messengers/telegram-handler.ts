import { Telegraf, Context } from 'telegraf';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { AgentService } from '@/services/agent-service';
import { MemoryService } from '@/services/memory-service';

export class TelegramHandler {
  private bot: Telegraf;
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

    if (!config.telegramToken) {
      throw new Error('TELEGRAM_BOT_TOKEN not configured');
    }

    this.bot = new Telegraf(config.telegramToken);
  }

  async start(): Promise<void> {
    this.setupHandlers();

    try {
      await this.bot.launch();
      this.logger.info('Telegram bot started');
    } catch (error) {
      this.logger.error('Failed to start Telegram bot', error as Error);
      throw error;
    }
  }

  private setupHandlers(): void {
    this.bot.start((ctx) => this.handleStart(ctx));
    this.bot.command('help', (ctx) => this.handleHelp(ctx));
    this.bot.command('search', (ctx) => this.handleSearch(ctx));
    this.bot.command('install', (ctx) => this.handleInstall(ctx));
    this.bot.command('history', (ctx) => this.handleHistory(ctx));
    this.bot.on('message', (ctx) => this.handleMessage(ctx));
  }

  private async handleStart(ctx: Context): Promise<void> {
    const userId = ctx.from?.id.toString();
    if (!userId) return;

    await this.memoryService.save({
      userId,
      key: 'telegram_user_name',
      value: ctx.from?.first_name || 'User',
    });

    await ctx.reply(
      '👋 Welcome to Hermes Agent!\n\n' +
        'I help you search and install NPM packages using natural language.\n\n' +
        'Try:\n' +
        '• /search express\n' +
        '• /install lodash\n' +
        '• /help for more options'
    );
  }

  private async handleHelp(ctx: Context): Promise<void> {
    await ctx.reply(
      '📚 Available Commands:\n\n' +
        '/search <query> - Search for NPM packages\n' +
        '/install <package> - Install an NPM package\n' +
        '/history - Show conversation history\n' +
        '/help - Show this help message\n\n' +
        'Or just chat naturally: "find packages for authentication"'
    );
  }

  private async handleSearch(ctx: Context): Promise<void> {
    const userId = ctx.from?.id.toString();
    if (!userId) return;

    const query = ctx.message?.text?.replace('/search', '').trim();

    if (!query) {
      await ctx.reply('Please provide a search query. Example: /search express');
      return;
    }

    await ctx.sendChatAction('typing');

    const conversationId = await this.memoryService.createConversation(
      userId,
      'telegram',
      `Search: ${query}`
    );

    try {
      const response = await this.agentService.processUserMessage({
        userId,
        platform: 'telegram',
        conversationId,
        userInput: `search ${query}`,
      });

      await this.memoryService.saveConversationMessage(
        conversationId,
        userId,
        'user',
        `search ${query}`
      );

      await this.memoryService.saveConversationMessage(
        conversationId,
        userId,
        'assistant',
        response.text
      );

      await ctx.reply(response.text, { parse_mode: 'Markdown' });
    } catch (error) {
      this.logger.error('Telegram search failed', error as Error);
      await ctx.reply('❌ Search failed. Please try again.');
    }
  }

  private async handleInstall(ctx: Context): Promise<void> {
    const userId = ctx.from?.id.toString();
    if (!userId) return;

    const packageName = ctx.message?.text?.replace('/install', '').trim();

    if (!packageName) {
      await ctx.reply('Please provide a package name. Example: /install express');
      return;
    }

    await ctx.sendChatAction('typing');

    const conversationId = await this.memoryService.createConversation(
      userId,
      'telegram',
      `Install: ${packageName}`
    );

    try {
      const response = await this.agentService.processUserMessage({
        userId,
        platform: 'telegram',
        conversationId,
        userInput: `install ${packageName}`,
      });

      await this.memoryService.saveConversationMessage(
        conversationId,
        userId,
        'assistant',
        response.text
      );

      await ctx.reply(response.text);
    } catch (error) {
      this.logger.error('Telegram install failed', error as Error);
      await ctx.reply('❌ Installation failed. Please try again.');
    }
  }

  private async handleHistory(ctx: Context): Promise<void> {
    const userId = ctx.from?.id.toString();
    if (!userId) return;

    const memories = await this.memoryService.retrieveAll(userId);

    if (Object.keys(memories).length === 0) {
      await ctx.reply('No conversation history found.');
      return;
    }

    let historyText = '📜 Your Activity:\n\n';
    for (const [key, value] of Object.entries(memories)) {
      historyText += `• ${key}: ${JSON.stringify(value).substring(0, 100)}...\n`;
    }

    await ctx.reply(historyText);
  }

  private async handleMessage(ctx: Context): Promise<void> {
    if (!ctx.message || !('text' in ctx.message)) return;

    const userId = ctx.from?.id.toString();
    if (!userId) return;

    const userInput = ctx.message.text;

    if (userInput.startsWith('/')) return;

    await ctx.sendChatAction('typing');

    const conversationId = await this.memoryService.createConversation(
      userId,
      'telegram'
    );

    try {
      const response = await this.agentService.processUserMessage({
        userId,
        platform: 'telegram',
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

      await ctx.reply(response.text, { parse_mode: 'Markdown' });
    } catch (error) {
      this.logger.error('Telegram message processing failed', error as Error);
      await ctx.reply('❌ Error processing your message. Please try again.');
    }
  }

  async stop(): Promise<void> {
    await this.bot.stop();
    this.logger.info('Telegram bot stopped');
  }
}
