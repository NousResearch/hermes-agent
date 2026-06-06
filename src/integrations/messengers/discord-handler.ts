import { Client, GatewayIntentBits, Message, ChannelType } from 'discord.js';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { AgentService } from '@/services/agent-service';
import { MemoryService } from '@/services/memory-service';

export class DiscordHandler {
  private client: Client;
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

    if (!config.discordToken) {
      throw new Error('DISCORD_BOT_TOKEN not configured');
    }

    this.client = new Client({
      intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.DirectMessages,
        GatewayIntentBits.MessageContent,
      ],
    });
  }

  async start(): Promise<void> {
    this.setupHandlers();

    try {
      await this.client.login(this.config.discordToken);
      this.logger.info('Discord bot connected');
    } catch (error) {
      this.logger.error('Failed to start Discord bot', error as Error);
      throw error;
    }
  }

  private setupHandlers(): void {
    this.client.on('ready', () => {
      this.logger.info(`Discord bot logged in as ${this.client.user?.tag}`);
    });

    this.client.on('messageCreate', (msg) => this.handleMessage(msg));
  }

  private async handleMessage(msg: Message): Promise<void> {
    if (msg.author.bot) return;

    const userId = msg.author.id;
    let conversationId: number;

    try {
      if (msg.channel.isDMBased()) {
        conversationId = await this.memoryService.createConversation(
          userId,
          'discord_dm'
        );
      } else {
        conversationId = await this.memoryService.createConversation(
          userId,
          'discord_guild'
        );
      }

      await msg.channel.sendTyping();

      const userInput = msg.content;

      if (userInput.startsWith('!help')) {
        await this.sendHelp(msg);
        return;
      }

      if (userInput.startsWith('!search')) {
        const query = userInput.replace('!search', '').trim();
        if (!query) {
          await msg.reply('Please provide a search query. Example: `!search express`');
          return;
        }

        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'discord',
          conversationId,
          userInput: `search ${query}`,
        });

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'assistant',
          response.text
        );

        await this.splitAndSendMessage(msg, response.text);
        return;
      }

      if (userInput.startsWith('!install')) {
        const packageName = userInput.replace('!install', '').trim();
        if (!packageName) {
          await msg.reply('Please provide a package name. Example: `!install express`');
          return;
        }

        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'discord',
          conversationId,
          userInput: `install ${packageName}`,
        });

        await this.memoryService.saveConversationMessage(
          conversationId,
          userId,
          'assistant',
          response.text
        );

        await msg.reply(response.text);
        return;
      }

      const response = await this.agentService.processUserMessage({
        userId,
        platform: 'discord',
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

      await this.splitAndSendMessage(msg, response.text);
    } catch (error) {
      this.logger.error('Discord message processing failed', error as Error);
      await msg.reply('❌ Error processing your message. Please try again.');
    }
  }

  private async sendHelp(msg: Message): Promise<void> {
    const helpText = `
🤖 **Hermes Agent - Help**

**Commands:**
\`!search <query>\` - Search for NPM packages
\`!install <package>\` - Install an NPM package
\`!help\` - Show this help message

**Examples:**
• \`!search authentication\`
• \`!install express@4.18.0\`

Or just chat naturally with me!
    `;

    await msg.reply(helpText);
  }

  private async splitAndSendMessage(msg: Message, text: string): Promise<void> {
    const MAX_LENGTH = 1900;
    const lines = text.split('\n');
    let currentMessage = '';

    for (const line of lines) {
      if ((currentMessage + line).length > MAX_LENGTH) {
        if (currentMessage.length > 0) {
          await msg.reply(currentMessage);
          currentMessage = '';
        }
      }
      currentMessage += line + '\n';
    }

    if (currentMessage.length > 0) {
      await msg.reply(currentMessage);
    }
  }

  async stop(): Promise<void> {
    await this.client.destroy();
    this.logger.info('Discord bot stopped');
  }
}
