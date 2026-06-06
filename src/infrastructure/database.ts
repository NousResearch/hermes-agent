import { Pool, PoolClient } from 'pg';
import { Logger } from './logger';
import { ConfigService } from './config';

export class DatabaseService {
  private pool: Pool;
  private readonly logger: Logger;
  private readonly config: ConfigService;

  constructor(config: ConfigService, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.pool = new Pool({
      connectionString: config.databaseUrl,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });
  }

  async initialize(): Promise<void> {
    try {
      const client = await this.pool.connect();
      this.logger.info('Database connection established');

      await this.createTables(client);
      client.release();

      this.pool.on('error', (err) => {
        this.logger.error('Unexpected error on idle client', err);
      });
    } catch (error) {
      this.logger.error('Failed to initialize database', error as Error);
      throw error;
    }
  }

  private async createTables(client: PoolClient): Promise<void> {
    const createTablesSQL = `
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) UNIQUE NOT NULL,
        platform VARCHAR(50) NOT NULL,
        name VARCHAR(255),
        email VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        platform VARCHAR(50) NOT NULL,
        title VARCHAR(255),
        context JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
      );

      CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
      );

      CREATE TABLE IF NOT EXISTS memory_entries (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        key VARCHAR(255) NOT NULL,
        value JSONB NOT NULL,
        ttl INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, key)
      );

      CREATE TABLE IF NOT EXISTS user_projects (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        project_id VARCHAR(255) NOT NULL,
        project_path TEXT NOT NULL,
        project_name VARCHAR(255),
        package_manager VARCHAR(20) DEFAULT 'npm',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, project_id)
      );

      CREATE TABLE IF NOT EXISTS package_installations (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(255) NOT NULL,
        user_id VARCHAR(255),
        package_name VARCHAR(255) NOT NULL,
        version VARCHAR(50),
        status VARCHAR(50) NOT NULL,
        error_message TEXT,
        duration_ms INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES user_projects(project_id)
      );

      CREATE TABLE IF NOT EXISTS npm_installations (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        package_name VARCHAR(255) NOT NULL,
        version VARCHAR(50),
        status VARCHAR(50) NOT NULL,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
      );

      CREATE TABLE IF NOT EXISTS mcp_calls (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        endpoint VARCHAR(500) NOT NULL,
        method VARCHAR(50) NOT NULL,
        payload JSONB,
        response JSONB,
        status INTEGER,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_users_platform ON users(platform);
      CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
      CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
      CREATE INDEX IF NOT EXISTS idx_memory_user_key ON memory_entries(user_id, key);
      CREATE INDEX IF NOT EXISTS idx_user_projects ON user_projects(user_id);
      CREATE INDEX IF NOT EXISTS idx_package_installations ON package_installations(project_id);
      CREATE INDEX IF NOT EXISTS idx_npm_installations_user ON npm_installations(user_id);
    `;

    await client.query(createTablesSQL);
    this.logger.info('Database tables initialized');
  }

  async query(sql: string, params?: any[]): Promise<any[]> {
    try {
      const result = await this.pool.query(sql, params);
      return result.rows;
    } catch (error) {
      this.logger.error('Database query failed', error as Error);
      throw error;
    }
  }

  async queryOne(sql: string, params?: any[]): Promise<any | null> {
    const results = await this.query(sql, params);
    return results[0] || null;
  }

  async execute(sql: string, params?: any[]): Promise<number> {
    try {
      const result = await this.pool.query(sql, params);
      return result.rowCount || 0;
    } catch (error) {
      this.logger.error('Database execute failed', error as Error);
      throw error;
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
    this.logger.info('Database connection closed');
  }
}
