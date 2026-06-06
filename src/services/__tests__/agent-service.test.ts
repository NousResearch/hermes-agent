import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AgentService } from '@/services/agent-service';
import { Logger } from '@/infrastructure/logger';

describe('AgentService - Intent Parsing', () => {
  let logger: Logger;

  beforeEach(() => {
    logger = new Logger('test');
  });

  it('should parse npm_search intent', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'search authentication libraries';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('npm_search');
    expect(intent.query).toContain('authentication');
  });

  it('should parse npm_install intent', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'install express@4.18.0';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('npm_install');
    expect(intent.packageName).toBe('express');
    expect(intent.version).toBe('4.18.0');
  });

  it('should parse project_register intent', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'register project "/home/user/myapp"';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('project_register');
    expect(intent.projectPath).toBe('/home/user/myapp');
  });

  it('should parse general intent for unknown input', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'what is npm?';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('general');
  });

  it('should handle add as install synonym', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'add lodash';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('npm_install');
    expect(intent.packageName).toBe('lodash');
  });

  it('should handle find as search synonym', () => {
    const agentService = new (AgentService as any)(
      {},
      {},
      {},
      {},
      {},
      logger
    );

    const input = 'find validation packages';
    const intent = agentService['parseIntent'](input);

    expect(intent.type).toBe('npm_search');
  });
});
