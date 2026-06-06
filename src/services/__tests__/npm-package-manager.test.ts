import { describe, it, expect, beforeEach, vi } from 'vitest';
import { NPMPackageManager } from '@/services/npm-package-manager';
import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';

describe('NPMPackageManager', () => {
  let npmManager: NPMPackageManager;
  let logger: Logger;
  let config: ConfigService;

  beforeEach(() => {
    logger = new Logger('test');
    config = new ConfigService();
    npmManager = new NPMPackageManager(config, logger);
  });

  it('should parse package strings correctly', () => {
    const result1 = npmManager.parsePackageString('express');
    expect(result1.name).toBe('express');
    expect(result1.version).toBeUndefined();

    const result2 = npmManager.parsePackageString('express@4.18.0');
    expect(result2.name).toBe('express');
    expect(result2.version).toBe('4.18.0');

    const result3 = npmManager.parsePackageString('@babel/core@7.22.0');
    expect(result3.name).toBe('@babel/core');
    expect(result3.version).toBe('7.22.0');
  });

  it('should throw error on invalid package string', () => {
    expect(() => npmManager.parsePackageString('')).toThrow();
    expect(() => npmManager.parsePackageString('   ')).toThrow();
  });

  it('should search packages from npm registry', async () => {
    const results = await npmManager.searchPackages('lodash', 5);
    
    expect(results).toHaveProperty('packages');
    expect(results).toHaveProperty('total');
    expect(Array.isArray(results.packages)).toBe(true);
    
    if (results.packages.length > 0) {
      const pkg = results.packages[0];
      expect(pkg).toHaveProperty('name');
      expect(pkg).toHaveProperty('version');
      expect(pkg).toHaveProperty('description');
    }
  });

  it('should validate existing package', async () => {
    const isValid = await npmManager.validatePackage('express');
    expect(isValid).toBe(true);
  });

  it('should return false for non-existing package', async () => {
    const isValid = await npmManager.validatePackage('this-package-definitely-does-not-exist-12345');
    expect(isValid).toBe(false);
  });

  it('should get package info', async () => {
    const info = await npmManager.getPackageInfo('express');
    expect(info).toBeDefined();
    expect(info.name).toBe('express');
  });

  it('should get package versions', async () => {
    const versions = await npmManager.getPackageVersions('express');
    expect(Array.isArray(versions)).toBe(true);
    expect(versions.length).toBeGreaterThan(0);
  });
});
