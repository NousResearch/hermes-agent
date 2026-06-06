import { Logger } from '@/infrastructure/logger';
import { DatabaseService } from '@/infrastructure/database';

export interface MemoryEntry {
  id?: string;
  userId: string;
  key: string;
  value: any;
  ttl?: number;
  createdAt?: Date;
  updatedAt?: Date;
}

export class MemoryService {
  private readonly database: DatabaseService;
  private readonly logger: Logger;
  private memoryCache: Map<string, any> = new Map();

  constructor(database: DatabaseService, logger: Logger) {
    this.database = database;
    this.logger = logger;
  }

  async save(entry: MemoryEntry): Promise<void> {
    try {
      const cacheKey = `${entry.userId}:${entry.key}`;

      if (entry.ttl && entry.ttl > 0) {
        setTimeout(() => {
          this.memoryCache.delete(cacheKey);
        }, entry.ttl * 1000);
      }

      this.memoryCache.set(cacheKey, entry.value);

      const sql = `
        INSERT INTO memory_entries (user_id, key, value, ttl)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, key)
        DO UPDATE SET value = $3, updated_at = CURRENT_TIMESTAMP
      `;

      await this.database.execute(sql, [
        entry.userId,
        entry.key,
        JSON.stringify(entry.value),
        entry.ttl,
      ]);

      this.logger.debug(`Memory saved: ${cacheKey}`);
    } catch (error) {
      this.logger.error('Failed to save memory', error as Error);
      throw error;
    }
  }

  async retrieve(userId: string, key: string): Promise<any | null> {
    try {
      const cacheKey = `${userId}:${key}`;

      if (this.memoryCache.has(cacheKey)) {
        return this.memoryCache.get(cacheKey);
      }

      const sql = `
        SELECT value FROM memory_entries
        WHERE user_id = $1 AND key = $2
      `;

      const result = await this.database.queryOne(sql, [userId, key]);

      if (result) {
        const value = JSON.parse(result.value);
        this.memoryCache.set(cacheKey, value);
        return value;
      }

      return null;
    } catch (error) {
      this.logger.error('Failed to retrieve memory', error as Error);
      throw error;
    }
  }

  async retrieveAll(userId: string): Promise<Record<string, any>> {
    try {
      const sql = `
        SELECT key, value FROM memory_entries
        WHERE user_id = $1
      `;

      const results = await this.database.query(sql, [userId]);

      const memory: Record<string, any> = {};
      for (const row of results) {
        memory[row.key] = JSON.parse(row.value);
      }

      return memory;
    } catch (error) {
      this.logger.error('Failed to retrieve all memories', error as Error);
      throw error;
    }
  }

  async delete(userId: string, key: string): Promise<void> {
    try {
      const cacheKey = `${userId}:${key}`;
      this.memoryCache.delete(cacheKey);

      const sql = `
        DELETE FROM memory_entries
        WHERE user_id = $1 AND key = $2
      `;

      await this.database.execute(sql, [userId, key]);

      this.logger.debug(`Memory deleted: ${cacheKey}`);
    } catch (error) {
      this.logger.error('Failed to delete memory', error as Error);
      throw error;
    }
  }

  async deleteAllForUser(userId: string): Promise<void> {
    try {
      const sql = `DELETE FROM memory_entries WHERE user_id = $1`;
      await this.database.execute(sql, [userId]);

      for (const key of this.memoryCache.keys()) {
        if (key.startsWith(`${userId}:`)) {
          this.memoryCache.delete(key);
        }
      }

      this.logger.debug(`All memories deleted for user: ${userId}`);
    } catch (error) {
      this.logger.error('Failed to delete user memories', error as Error);
      throw error;
    }
  }

  async saveConversationMessage(
    conversationId: number,
    userId: string,
    role: 'user' | 'assistant',
    content: string,
    metadata?: Record<string, any>
  ): Promise<void> {
    try {
      const sql = `
        INSERT INTO messages (conversation_id, user_id, role, content, metadata)
        VALUES ($1, $2, $3, $4, $5)
      `;

      await this.database.execute(sql, [
        conversationId,
        userId,
        role,
        content,
        JSON.stringify(metadata || {}),
      ]);

      this.logger.debug(`Message saved: ${conversationId}`);
    } catch (error) {
      this.logger.error('Failed to save message', error as Error);
      throw error;
    }
  }

  async getConversationHistory(
    conversationId: number,
    limit: number = 50
  ): Promise<any[]> {
    try {
      const sql = `
        SELECT role, content, metadata, created_at FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at DESC
        LIMIT $2
      `;

      const messages = await this.database.query(sql, [conversationId, limit]);
      return messages.reverse();
    } catch (error) {
      this.logger.error('Failed to retrieve conversation history', error as Error);
      throw error;
    }
  }

  async createConversation(
    userId: string,
    platform: string,
    title?: string
  ): Promise<number> {
    try {
      const sql = `
        INSERT INTO conversations (user_id, platform, title, context)
        VALUES ($1, $2, $3, $4)
        RETURNING id
      `;

      const result = await this.database.queryOne(sql, [userId, platform, title, '{}']);
      return result.id;
    } catch (error) {
      this.logger.error('Failed to create conversation', error as Error);
      throw error;
    }
  }
}
