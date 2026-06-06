import { Client } from 'whatsapp-web.js';
import * as qrcode from 'qrcode-terminal';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { AgentService } from '@/services/agent-service';
import { MemoryService } from '@/services/memory-service';

export class WhatsAppHandler {
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

    this.client = new Client();
  }

  async start(): Promise<void> {
    this.setupHandlers();

    try {
      await this.client.initialize();
      this.logger.info('WhatsApp client initialized');
    } catch (error) {
      this.logger.error('Failed to initialize WhatsApp client', error as Error);
      throw error;
    }
  }

  private setupHandlers(): void {
    this.client.on('qr', (qr) => {
      this.logger.info('Scan this QR code with WhatsApp:');
      qrcode.generate(qr, { small: true });
    });

    this.client.on('ready', () => {
      this.logger.info('WhatsApp client ready');
    });

    this.client.on('message', (msg) => this.handleMessage(msg));
  }

  private async handleMessage(msg: any): Promise<void> {
    if (msg.from === 'status@broadcast') return;

    const userId = msg.from;
    let conversationId: number;

    try {
      conversationId = await this.memoryService.createConversation(
        userId,
        'whatsapp'
      );

      const userInput = msg.body;

      if (userInput.toLowerCase().startsWith('help')) {
        await this.sendHelp(msg);
        return;
      }

      if (userInput.toLowerCase().startsWith('search')) {
        const query = userInput.replace(/search/i, '').trim();
        if (!query) {
          await msg.reply('Please provide a search query. Example: search express');
          return;
        }

        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'whatsapp',
          conversationId,
          userInput: `search ${query}`,
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

      if (userInput.toLowerCase().startsWith('install')) {
        const packageName = userInput.replace(/install/i, '').trim();
        if (!packageName) {
          await msg.reply('Please provide a package name. Example: install express');
          return;
        }

        const response = await this.agentService.processUserMessage({
          userId,
          platform: 'whatsapp',
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
        platform: 'whatsapp',
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
      this.logger.error('WhatsApp message processing failed', error as Error);
      await msg.reply('❌ Error processing your message. Please try again.');
    }
  }

  private async sendHelp(msg: any): Promise<void> {
    const helpText = `
🤖 *Hermes Agent - Help*

*Commands:*
search <query> - Search for NPM packages
install <package> - Install an NPM package
help - Show this help message

*Examples:*
• search authentication
• install express@4.18.0

Or just chat naturally with me!
    `;

    await msg.reply(helpText);
  }

  private async splitAndSendMessage(msg: any, text: string): Promise<void> {
    const MAX_LENGTH = 4000;

    if (text.length <= MAX_LENGTH) {
      await msg.reply(text);
      return;
    }

    const chunks = text.match(new RegExp(`.{1,${MAX_LENGTH}}`, 'g')) || [];
    for (const chunk of chunks) {
      await msg.reply(chunk);
    }
  }

  async stop(): Promise<void> {
    await this.client.destroy();
    this.logger.info('WhatsApp client stopped');
  }
}
