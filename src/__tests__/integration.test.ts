import { describe, it, expect } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import { TEST_PROJECT_PATH, TEST_PACKAGE_NAME } from './config';

describe('Integration Tests - NPM Package Installation', () => {
  beforeAll(async () => {
    try {
      await fs.mkdir(TEST_PROJECT_PATH, { recursive: true });
      
      const packageJson = {
        name: 'test-hermes-project',
        version: '1.0.0',
        description: 'Test project for Hermes Agent',
        main: 'index.js',
        scripts: {
          test: 'echo "Error: no test specified" && exit 1',
        },
        keywords: [],
        author: '',
        license: 'ISC',
        dependencies: {},
      };

      await fs.writeFile(
        path.join(TEST_PROJECT_PATH, 'package.json'),
        JSON.stringify(packageJson, null, 2)
      );

      console.log(`Test project created at: ${TEST_PROJECT_PATH}`);
    } catch (error) {
      console.error('Failed to create test project:', error);
    }
  });

  afterAll(async () => {
    try {
      await fs.rm(TEST_PROJECT_PATH, { recursive: true, force: true });
      console.log('Test project cleaned up');
    } catch (error) {
      console.error('Failed to clean up test project:', error);
    }
  });

  it('should read package.json from test project', async () => {
    const packageJsonPath = path.join(TEST_PROJECT_PATH, 'package.json');
    const content = await fs.readFile(packageJsonPath, 'utf-8');
    const packageJson = JSON.parse(content);

    expect(packageJson.name).toBe('test-hermes-project');
    expect(packageJson).toHaveProperty('dependencies');
  });

  it('should verify test project structure', async () => {
    const stats = await fs.stat(TEST_PROJECT_PATH);
    expect(stats.isDirectory()).toBe(true);

    const packageJsonPath = path.join(TEST_PROJECT_PATH, 'package.json');
    const stats2 = await fs.stat(packageJsonPath);
    expect(stats2.isFile()).toBe(true);
  });

  it('should have writable permissions in test project', async () => {
    const testFile = path.join(TEST_PROJECT_PATH, '.test-write');
    
    await fs.writeFile(testFile, 'test');
    const content = await fs.readFile(testFile, 'utf-8');
    expect(content).toBe('test');
    
    await fs.rm(testFile);
  });
});
