import { createClient, RedisClientType } from 'redis';
import { Logger } from './logger';
import { ConfigService } from './config';

export class RedisService {
  private client: RedisClientType;
  private readonly logger: Logger;
  private readonly config: ConfigService;

  constructor(config: ConfigService, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.client = createClient({
      url: config.redisUrl,
      socket: {
        reconnectStrategy: (retries) => Math.min(retries * 50, 500),
      },
    });

    this.client.on('error', (err) => this.logger.error('Redis error', err));
    this.client.on('connect', () => this.logger.info('Redis connected'));
  }

  async connect(): Promise<void> {
    try {
      await this.client.connect();
      this.logger.info('Redis connection established');
    } catch (error) {
      this.logger.error('Failed to connect to Redis', error as Error);
      throw error;
    }
  }

  async get(key: string): Promise<string | null> {
    return await this.client.get(key);
  }

  async getJson(key: string): Promise<any> {
    const data = await this.client.get(key);
    return data ? JSON.parse(data) : null;
  }

  async set(key: string, value: string, ttlSeconds?: number): Promise<void> {
    if (ttlSeconds) {
      await this.client.setEx(key, ttlSeconds, value);
    } else {
      await this.client.set(key, value);
    }
  }

  async setJson(key: string, value: any, ttlSeconds?: number): Promise<void> {
    await this.set(key, JSON.stringify(value), ttlSeconds);
  }

  async del(key: string): Promise<number> {
    return await this.client.del(key);
  }

  async exists(key: string): Promise<boolean> {
    return (await this.client.exists(key)) > 0;
  }

  async incr(key: string): Promise<number> {
    return await this.client.incr(key);
  }

  async lpush(key: string, value: string): Promise<number> {
    return await this.client.lPush(key, value);
  }

  async lrange(key: string, start: number, stop: number): Promise<string[]> {
    return await this.client.lRange(key, start, stop);
  }

  async hset(key: string, field: string, value: string): Promise<number> {
    return await this.client.hSet(key, field, value);
  }

  async hget(key: string, field: string): Promise<string | null> {
    return await this.client.hGet(key, field);
  }

  async hgetall(key: string): Promise<Record<string, string>> {
    return await this.client.hGetAll(key);
  }

  async close(): Promise<void> {
    await this.client.quit();
    this.logger.info('Redis connection closed');
  }
}
