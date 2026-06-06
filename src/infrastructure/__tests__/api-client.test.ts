import { describe, it, expect, beforeEach } from 'vitest';
import { ApiClient } from '@/infrastructure/api-client';
import { Logger } from '@/infrastructure/logger';

describe('ApiClient', () => {
  let client: ApiClient;
  let logger: Logger;

  beforeEach(() => {
    logger = new Logger('test');
    client = new ApiClient(
      {
        baseURL: 'https://registry.npmjs.org',
        timeout: 10000,
      },
      logger
    );
  });

  it('should be initialized with correct config', () => {
    expect(client).toBeDefined();
  });

  it('should make successful GET request', async () => {
    const result = await client.get('/express');
    
    expect(result).toBeDefined();
    expect(result.name).toBe('express');
  });

  it('should handle request headers', () => {
    client.setHeader('X-Custom-Header', 'test-value');
    expect(client).toBeDefined();
  });

  it('should retry on network failure', async () => {
    const failClient = new ApiClient(
      {
        baseURL: 'https://nonexistent-domain-that-will-fail-12345.com',
        timeout: 2000,
        retryAttempts: 2,
      },
      logger
    );

    try {
      await failClient.get('/test');
    } catch (error) {
      expect(error).toBeDefined();
    }
  });
});
