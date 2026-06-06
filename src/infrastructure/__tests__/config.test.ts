import { describe, it, expect, beforeEach } from 'vitest';
import { ConfigService } from '@/infrastructure/config';

describe('ConfigService', () => {
  beforeEach(() => {
    process.env.NODE_ENV = 'test';
    process.env.PORT = '3000';
    process.env.LOG_LEVEL = 'debug';
    process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test';
    process.env.REDIS_URL = 'redis://localhost:6379';
  });

  it('should load environment variables', () => {
    const config = new ConfigService();

    expect(config.port).toBe(3000);
    expect(config.nodeEnv).toBe('test');
    expect(config.logLevel).toBe('debug');
  });

  it('should detect development environment', () => {
    process.env.NODE_ENV = 'development';
    const config = new ConfigService();

    expect(config.isDevelopment()).toBe(true);
    expect(config.isProduction()).toBe(false);
  });

  it('should detect production environment', () => {
    process.env.NODE_ENV = 'production';
    const config = new ConfigService();

    expect(config.isProduction()).toBe(true);
    expect(config.isDevelopment()).toBe(false);
  });

  it('should parse MCP endpoints as array', () => {
    process.env.MCP_ENDPOINTS = 'http://localhost:3001,http://example.com:8080';
    const config = new ConfigService();

    expect(Array.isArray(config.mcpEndpoints)).toBe(true);
    expect(config.mcpEndpoints.length).toBe(2);
    expect(config.mcpEndpoints[0]).toBe('http://localhost:3001');
  });

  it('should return empty array for missing MCP endpoints', () => {
    delete process.env.MCP_ENDPOINTS;
    const config = new ConfigService();

    expect(Array.isArray(config.mcpEndpoints)).toBe(true);
    expect(config.mcpEndpoints.length).toBe(0);
  });

  it('should throw error for missing required environment variables', () => {
    delete process.env.DATABASE_URL;

    expect(() => new ConfigService()).toThrow();
  });
});
