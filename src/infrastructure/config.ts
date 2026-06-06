import { Logger } from '@/infrastructure/logger';

export class ConfigService {
  private readonly env = process.env;

  constructor() {
    this.validateRequiredEnvVars();
  }

  private validateRequiredEnvVars(): void {
    const required = ['DATABASE_URL', 'REDIS_URL'];
    const missing = required.filter((key) => !this.env[key]);

    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
  }

  get nodeEnv(): string {
    return this.env.NODE_ENV || 'development';
  }

  get port(): number {
    return parseInt(this.env.PORT || '3000', 10);
  }

  get logLevel(): string {
    return this.env.LOG_LEVEL || 'info';
  }

  get databaseUrl(): string {
    return this.env.DATABASE_URL || '';
  }

  get redisUrl(): string {
    return this.env.REDIS_URL || '';
  }

  get telegramToken(): string | undefined {
    return this.env.TELEGRAM_BOT_TOKEN;
  }

  get discordToken(): string | undefined {
    return this.env.DISCORD_BOT_TOKEN;
  }

  get slackBotToken(): string | undefined {
    return this.env.SLACK_BOT_TOKEN;
  }

  get slackSigningSecret(): string | undefined {
    return this.env.SLACK_SIGNING_SECRET;
  }

  get whatsappAccountSid(): string | undefined {
    return this.env.WHATSAPP_ACCOUNT_SID;
  }

  get whatsappAuthToken(): string | undefined {
    return this.env.WHATSAPP_AUTH_TOKEN;
  }

  get mcpEndpoints(): string[] {
    return (this.env.MCP_ENDPOINTS || '').split(',').filter(Boolean);
  }

  get mcpApiKey(): string | undefined {
    return this.env.MCP_API_KEY;
  }

  get npmRegistryUrl(): string {
    return this.env.NPM_REGISTRY_URL || 'https://registry.npmjs.org';
  }

  get npmToken(): string | undefined {
    return this.env.NPM_TOKEN;
  }

  get openaiApiKey(): string | undefined {
    return this.env.OPENAI_API_KEY;
  }

  get anthropicApiKey(): string | undefined {
    return this.env.ANTHROPIC_API_KEY;
  }

  get jwtSecret(): string {
    return this.env.JWT_SECRET || 'dev-secret-key';
  }

  get encryptionKey(): string {
    return this.env.ENCRYPTION_KEY || 'dev-encryption-key-32-chars-long!';
  }

  get rateLimitWindowMs(): number {
    return parseInt(this.env.RATE_LIMIT_WINDOW_MS || '900000', 10);
  }

  get rateLimitMaxRequests(): number {
    return parseInt(this.env.RATE_LIMIT_MAX_REQUESTS || '100', 10);
  }

  isProduction(): boolean {
    return this.nodeEnv === 'production';
  }

  isDevelopment(): boolean {
    return this.nodeEnv === 'development';
  }
}
