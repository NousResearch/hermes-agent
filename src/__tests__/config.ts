export const NODE_ENV = process.env.NODE_ENV || 'test';
export const PORT = process.env.PORT || '3000';
export const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://test:[REDACTED]@localhost:5432/hermes_test';
export const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

export const TEST_PROJECT_PATH = '/tmp/test-hermes-project';
export const TEST_USER_ID = 'test-user-123';
export const TEST_PACKAGE_NAME = 'lodash';
export const TEST_PACKAGE_VERSION = '4.17.21';
