import { describe, it, expect } from 'vitest';
import { Logger } from '@/infrastructure/logger';

describe('Logger', () => {
  let logger: Logger;

  beforeEach(() => {
    logger = new Logger('test-context');
  });

  it('should create logger instance', () => {
    expect(logger).toBeDefined();
  });

  it('should log debug messages', () => {
    expect(() => {
      logger.debug('Test debug message', { key: 'value' });
    }).not.toThrow();
  });

  it('should log info messages', () => {
    expect(() => {
      logger.info('Test info message', { status: 'ok' });
    }).not.toThrow();
  });

  it('should log warning messages', () => {
    expect(() => {
      logger.warn('Test warning message', { warning: true });
    }).not.toThrow();
  });

  it('should log error messages', () => {
    const error = new Error('Test error');
    expect(() => {
      logger.error('Test error message', error);
    }).not.toThrow();
  });

  it('should handle error objects', () => {
    const error = new Error('Detailed error message');
    expect(() => {
      logger.error('An error occurred', error);
    }).not.toThrow();
  });
});
