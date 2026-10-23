#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { chmodSync, cpSync, existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const skillSourceDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const hermesHome = process.env.HERMES_HOME || join(homedir(), '.hermes');
const hermesCommand = process.env.SENDIT_HERMES_COMMAND || 'hermes';
const configPath = join(hermesHome, 'config.yaml');
const skillTargetDir = join(hermesHome, 'skills', 'social-media', 'sendit');
const standardMcpUrl = 'https://sendit.infiniteappsai.com/api/mcp';

/** Use Hermes's YAML writer so inline mappings, comments, and auth blocks stay valid. */
function configureSendIt() {
  mkdirSync(hermesHome, { recursive: true });
  const original = existsSync(configPath) ? readFileSync(configPath) : null;
  const backupPath = `${configPath}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  if (original !== null) {
    writeFileSync(backupPath, original, { mode: 0o600 });
  }

  try {
    for (const [key, value] of [
      ['mcp_servers.sendit.url', standardMcpUrl],
      ['mcp_servers.sendit.auth', 'oauth'],
    ]) {
      const result = spawnSync(hermesCommand, ['config', 'set', '--force', key, value], {
        env: { ...process.env, HERMES_HOME: hermesHome },
        encoding: 'utf8',
        timeout: 60000,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      if (result.error || result.status !== 0) {
        // Do not print config contents or subprocess output, which may include credentials.
        throw new Error(`Hermes could not set ${key}. Check your Hermes installation and run hermes update.`);
      }
    }
    if (!existsSync(configPath)) {
      throw new Error('Hermes reported success but did not create config.yaml.');
    }
    chmodSync(configPath, 0o600);
  } catch (error) {
    if (original !== null) {
      writeFileSync(configPath, original);
    } else {
      rmSync(configPath, { force: true });
    }
    throw error;
  }
  console.log(`Configured SendIt remote OAuth MCP: ${configPath}`);
}

function installSkill() {
  const skillsRoot = resolve(hermesHome, 'skills');
  const sourceWithinSkills = relative(
    existsSync(skillsRoot) ? realpathSync(skillsRoot) : skillsRoot,
    realpathSync(skillSourceDir)
  );
  if (!sourceWithinSkills.startsWith('..') && !isAbsolute(sourceWithinSkills)) {
    console.log(`SendIt skill already installed: ${skillSourceDir}`);
    return;
  }
  mkdirSync(dirname(skillTargetDir), { recursive: true });
  cpSync(skillSourceDir, skillTargetDir, {
    recursive: true,
    force: true,
    filter: (src) => !src.includes('/.DS_Store'),
  });
  console.log(`Installed SendIt skill: ${skillTargetDir}`);
}

try {
  configureSendIt();
  installSkill();
  console.log('Next: run hermes mcp login sendit, then /reload-mcp in your session.');
  console.log('For Telegram or a headless VPS, follow references/remote-oauth.md.');
} catch (error) {
  console.error(`[SendIt] ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
