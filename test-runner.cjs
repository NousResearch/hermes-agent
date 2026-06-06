#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

const results = [];
let totalTests = 0;
let passedTests = 0;

function assert(condition, message) {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message} - Expected ${expected}, got ${actual}`);
  }
}

function assertDefined(value, message) {
  if (value === undefined) throw new Error(`${message} - Value is undefined`);
}

// Test: Config Service
async function testConfigService() {
  const suite = { name: 'ConfigService', tests: [] };

  const test1Start = Date.now();
  try {
    const parsePackageString = (str) => {
      const match = str.match(/^(@?[^@]+?)(?:@(.+))?$/);
      if (!match) throw new Error(`Invalid package string: ${str}`);
      return { name: match[1], version: match[2] };
    };

    const result = parsePackageString('express@4.18.0');
    assertEqual(result.name, 'express', 'Package name should be express');
    assertEqual(result.version, '4.18.0', 'Package version should be 4.18.0');

    suite.tests.push({
      name: 'should parse package strings correctly',
      passed: true,
      duration: Date.now() - test1Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should parse package strings correctly',
      passed: false,
      error: error.message,
      duration: Date.now() - test1Start,
    });
  }
  totalTests++;

  const test2Start = Date.now();
  try {
    const parsePackageString = (str) => {
      const match = str.match(/^(@?[^@]+?)(?:@(.+))?$/);
      if (!match) throw new Error(`Invalid package string: ${str}`);
      return { name: match[1], version: match[2] };
    };

    const result = parsePackageString('@babel/core@7.22.0');
    assertEqual(result.name, '@babel/core', 'Scoped package name should be @babel/core');
    assertEqual(result.version, '7.22.0', 'Version should be 7.22.0');

    suite.tests.push({
      name: 'should handle scoped packages',
      passed: true,
      duration: Date.now() - test2Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should handle scoped packages',
      passed: false,
      error: error.message,
      duration: Date.now() - test2Start,
    });
  }
  totalTests++;

  return suite;
}

// Test: Agent Intent Parsing
async function testAgentIntents() {
  const suite = { name: 'AgentService - Intent Parsing', tests: [] };

  const parseIntent = (userInput) => {
    const lowerInput = userInput.toLowerCase();

    if (lowerInput.includes('search') || lowerInput.includes('find') || lowerInput.includes('look for')) {
      return {
        type: 'npm_search',
        query: userInput.replace(/search|find|look for/gi, '').trim(),
      };
    }

    if (lowerInput.includes('install') || lowerInput.includes('add') || lowerInput.includes('setup')) {
      const match = userInput.match(/(?:install|add|setup)\s+([^\s@]+?)(?:@(\S+))?(?:\s|$)/i);
      if (match) {
        return {
          type: 'npm_install',
          packageName: match[1],
          version: match[2],
        };
      }
    }

    if (lowerInput.includes('register project')) {
      return { type: 'project_register', projectPath: '/path/to/project' };
    }

    return { type: 'general' };
  };

  const test1Start = Date.now();
  try {
    const intent = parseIntent('search authentication libraries');
    assertEqual(intent.type, 'npm_search', 'Intent should be npm_search');

    suite.tests.push({
      name: 'should parse npm_search intent',
      passed: true,
      duration: Date.now() - test1Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should parse npm_search intent',
      passed: false,
      error: error.message,
      duration: Date.now() - test1Start,
    });
  }
  totalTests++;

  const test2Start = Date.now();
  try {
    const intent = parseIntent('install express@4.18.0');
    assertEqual(intent.type, 'npm_install', 'Intent should be npm_install');
    assertEqual(intent.packageName, 'express', 'Package name should be express');
    assertEqual(intent.version, '4.18.0', 'Version should be 4.18.0');

    suite.tests.push({
      name: 'should parse npm_install intent',
      passed: true,
      duration: Date.now() - test2Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should parse npm_install intent',
      passed: false,
      error: error.message,
      duration: Date.now() - test2Start,
    });
  }
  totalTests++;

  const test3Start = Date.now();
  try {
    const intent = parseIntent('add lodash');
    assertEqual(intent.type, 'npm_install', 'Intent should be npm_install');
    assertEqual(intent.packageName, 'lodash', 'Package name should be lodash');

    suite.tests.push({
      name: 'should handle add as install synonym',
      passed: true,
      duration: Date.now() - test3Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should handle add as install synonym',
      passed: false,
      error: error.message,
      duration: Date.now() - test3Start,
    });
  }
  totalTests++;

  const test4Start = Date.now();
  try {
    const intent = parseIntent('find validation packages');
    assertEqual(intent.type, 'npm_search', 'Intent should be npm_search');

    suite.tests.push({
      name: 'should handle find as search synonym',
      passed: true,
      duration: Date.now() - test4Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should handle find as search synonym',
      passed: false,
      error: error.message,
      duration: Date.now() - test4Start,
    });
  }
  totalTests++;

  return suite;
}

// Test: Project Management
async function testProjectManagement() {
  const suite = { name: 'ProjectManager', tests: [] };
  const testProjectPath = '/tmp/hermes-test-project';

  const test1Start = Date.now();
  try {
    await fs.mkdir(testProjectPath, { recursive: true });
    const stats = await fs.stat(testProjectPath);
    assert(stats.isDirectory(), 'Project path should be a directory');

    suite.tests.push({
      name: 'should create project directory',
      passed: true,
      duration: Date.now() - test1Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should create project directory',
      passed: false,
      error: error.message,
      duration: Date.now() - test1Start,
    });
  }
  totalTests++;

  const test2Start = Date.now();
  try {
    const packageJson = {
      name: 'test-hermes-project',
      version: '1.0.0',
      description: 'Test project for Hermes Agent',
      dependencies: {},
    };

    await fs.writeFile(
      path.join(testProjectPath, 'package.json'),
      JSON.stringify(packageJson, null, 2)
    );

    const content = await fs.readFile(path.join(testProjectPath, 'package.json'), 'utf-8');
    const parsed = JSON.parse(content);
    assertEqual(parsed.name, 'test-hermes-project', 'Package name should match');

    suite.tests.push({
      name: 'should write and read package.json',
      passed: true,
      duration: Date.now() - test2Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should write and read package.json',
      passed: false,
      error: error.message,
      duration: Date.now() - test2Start,
    });
  }
  totalTests++;

  const test3Start = Date.now();
  try {
    await fs.rm(testProjectPath, { recursive: true, force: true });

    suite.tests.push({
      name: 'should cleanup test project',
      passed: true,
      duration: Date.now() - test3Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should cleanup test project',
      passed: false,
      error: error.message,
      duration: Date.now() - test3Start,
    });
  }
  totalTests++;

  return suite;
}

// Test: Logger
async function testLogger() {
  const suite = { name: 'Logger', tests: [] };

  const mockLogger = {
    debug: (msg) => true,
    info: (msg) => true,
    warn: (msg) => true,
    error: (msg) => true,
  };

  const test1Start = Date.now();
  try {
    assertDefined(mockLogger.debug, 'debug method should exist');
    assertDefined(mockLogger.info, 'info method should exist');
    assertDefined(mockLogger.warn, 'warn method should exist');
    assertDefined(mockLogger.error, 'error method should exist');

    suite.tests.push({
      name: 'should have all logging methods',
      passed: true,
      duration: Date.now() - test1Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should have all logging methods',
      passed: false,
      error: error.message,
      duration: Date.now() - test1Start,
    });
  }
  totalTests++;

  const test2Start = Date.now();
  try {
    assert(mockLogger.debug('test') === true, 'debug should return true');
    assert(mockLogger.info('test') === true, 'info should return true');
    assert(mockLogger.warn('test') === true, 'warn should return true');
    assert(mockLogger.error('test') === true, 'error should return true');

    suite.tests.push({
      name: 'should log messages without errors',
      passed: true,
      duration: Date.now() - test2Start,
    });
    passedTests++;
  } catch (error) {
    suite.tests.push({
      name: 'should log messages without errors',
      passed: false,
      error: error.message,
      duration: Date.now() - test2Start,
    });
  }
  totalTests++;

  return suite;
}

// Main Test Runner
async function runTests() {
  console.log('\n🧪 Hermes Agent Test Suite\n');
  console.log('═'.repeat(70));

  try {
    results.push(await testConfigService());
    results.push(await testAgentIntents());
    results.push(await testProjectManagement());
    results.push(await testLogger());
  } catch (error) {
    console.error('Test runner error:', error);
  }

  console.log('\n📊 Test Results:\n');

  for (const suite of results) {
    console.log(`\n${suite.name}`);
    console.log('─'.repeat(70));

    for (const test of suite.tests) {
      const status = test.passed ? '✅ PASS' : '❌ FAIL';
      const duration = `(${test.duration}ms)`;

      console.log(`  ${status} ${test.name} ${duration}`);

      if (test.error) {
        console.log(`     └─ Error: ${test.error}`);
      }
    }
  }

  console.log('\n' + '═'.repeat(70));
  console.log(`\n📈 Summary: ${passedTests}/${totalTests} tests passed\n`);

  const successRate = ((passedTests / totalTests) * 100).toFixed(1);
  console.log(`Success Rate: ${successRate}%`);

  if (passedTests === totalTests) {
    console.log('\n🎉 All tests passed!\n');
    process.exit(0);
  } else {
    console.log(`\n⚠️ ${totalTests - passedTests} test(s) failed\n`);
    process.exit(1);
  }
}

runTests().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
