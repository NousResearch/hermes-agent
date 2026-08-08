#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const skillSourceDir = resolve(scriptDir, '..');
const hermesHome = process.env.HERMES_HOME || join(homedir(), '.hermes');
const configPath = join(hermesHome, 'config.yaml');
const skillTargetDir = join(hermesHome, 'skills', 'social-media', 'sendit');
const standardMcpUrl = 'https://sendit.infiniteappsai.com/api/mcp';

const senditBlock = [
  '  sendit:',
  `    url: "${standardMcpUrl}"`,
  '    auth: oauth',
].join('\n');

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function backupConfig(original) {
  writeFileSync(`${configPath}.bak-${timestamp()}`, original, 'utf8');
}

function installSkill() {
  if (resolve(skillSourceDir) === resolve(skillTargetDir)) {
    console.log(`SendIt skill already running from install target: ${skillTargetDir}`);
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

function findMcpSection(lines) {
  const start = lines.findIndex((line) => /^mcp_servers:\s*(#.*)?$/.test(line));
  if (start === -1) return null;

  let end = lines.length;
  for (let i = start + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.trim() === '' || line.trimStart().startsWith('#')) continue;
    if (!/^[ \t]/.test(line)) {
      end = i;
      break;
    }
  }

  return { start, end };
}

function findSendItBlock(lines, mcpSection) {
  for (let i = mcpSection.start + 1; i < mcpSection.end; i += 1) {
    if (/^[ \t]{2}sendit:\s*(#.*)?$/.test(lines[i])) {
      let end = mcpSection.end;
      for (let j = i + 1; j < mcpSection.end; j += 1) {
        const line = lines[j];
        if (line.trim() === '' || line.trimStart().startsWith('#')) continue;
        if (/^[ \t]{2}\S/.test(line)) {
          end = j;
          break;
        }
      }
      return { start: i, end };
    }
  }

  return null;
}

function repairSendItServer(text) {
  const lines = text.split(/\r?\n/);
  const mcpSection = findMcpSection(lines);
  if (!mcpSection) return null;

  const senditSection = findSendItBlock(lines, mcpSection);
  if (!senditSection) return null;

  const existingBlock = lines.slice(senditSection.start, senditSection.end);
  const urlIndex = existingBlock.findIndex((line) => /^[ \t]{4}url:\s*/.test(line));
  const authIndex = existingBlock.findIndex((line) => /^[ \t]{4}auth:\s*/.test(line));
  let changed = false;

  if (urlIndex === -1) {
    existingBlock.splice(1, 0, `    url: "${standardMcpUrl}"`);
    changed = true;
  } else if (!existingBlock[urlIndex].includes(standardMcpUrl)) {
    existingBlock[urlIndex] = `    url: "${standardMcpUrl}"`;
    changed = true;
  }

  if (authIndex === -1) {
    existingBlock.push('    auth: oauth');
    changed = true;
  } else if (!/auth:\s*oauth\s*(#.*)?$/.test(existingBlock[authIndex])) {
    existingBlock[authIndex] = '    auth: oauth';
    changed = true;
  }

  if (!changed) return text;

  lines.splice(senditSection.start, senditSection.end - senditSection.start, ...existingBlock);
  return lines.join('\n').replace(/\s*$/, '\n');
}

function addMcpConfig() {
  mkdirSync(hermesHome, { recursive: true });

  if (!existsSync(configPath)) {
    writeFileSync(configPath, `mcp_servers:\n${senditBlock}\n`, 'utf8');
    console.log(`Created Hermes config with SendIt MCP server: ${configPath}`);
    return;
  }

  const original = readFileSync(configPath, 'utf8');
  const repaired = repairSendItServer(original);
  if (repaired) {
    if (repaired !== original) {
      backupConfig(original);
      writeFileSync(configPath, repaired, 'utf8');
      console.log(`Updated existing SendIt MCP server to standard endpoint: ${standardMcpUrl}`);
    } else {
      console.log('SendIt MCP server already exists in config.yaml.');
    }
    return;
  }

  backupConfig(original);

  const lines = original.split(/\r?\n/);
  const mcpSection = findMcpSection(lines);

  if (!mcpSection) {
    const next = `${original.replace(/\s*$/, '')}\n\nmcp_servers:\n${senditBlock}\n`;
    writeFileSync(configPath, next, 'utf8');
    console.log('Added mcp_servers.sendit to config.yaml.');
    return;
  }

  lines.splice(mcpSection.end, 0, ...senditBlock.split('\n'));
  writeFileSync(configPath, lines.join('\n').replace(/\s*$/, '\n'), 'utf8');
  console.log('Added mcp_servers.sendit to config.yaml.');
}

installSkill();
addMcpConfig();

console.log('');
console.log('Next: run `node scripts/start-oauth-login.mjs` from this setup folder.');
