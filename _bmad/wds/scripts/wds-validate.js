// wds-validate.js — WDS scaffold: validate page spec files for correctness
// Usage: node src/scripts/wds-validate.js --page "C-UX-Scenarios/01-onboarding/01-start/01-start.md"
//        node src/scripts/wds-validate.js --scenario "01 Onboarding"
//        node src/scripts/wds-validate.js --all

'use strict';

const fs = require('node:fs');
const path = require('node:path');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const value = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : true;
      args[key] = value;
      if (value !== true) i++;
    }
  }
  return args;
}

function toSlug(str) {
  return str.toLowerCase().replaceAll(/\s+/g, '-');
}

function printUsage() {
  process.stdout.write(
    [
      'Usage: node src/scripts/wds-validate.js --page <path>',
      '       node src/scripts/wds-validate.js --scenario "01 Onboarding"',
      '       node src/scripts/wds-validate.js --all',
      '',
      'Options:',
      '  --page        Path to a single page spec .md file',
      '  --scenario    Scenario name or slug to validate all pages',
      '  --all         Validate all scenarios',
      '  --output      Base path (default: current directory)',
      '',
    ].join('\n'),
  );
}

const REQUIRED_SECTIONS = [
  '## Overview',
  '## Page Metadata',
  '## Layout Structure',
  '## Spacing',
  '## Typography',
  '## Page Sections',
  '## Page States',
  '## Checklist',
];

const REQUIRED_METADATA_PROPS = ['Scenario', 'Page Number', 'Platform', 'Page Type', 'Viewport', 'Interaction', 'Visibility'];

const KEBAB_CASE_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

// Extract all OBJECT IDs from content: lines matching **OBJECT ID:** `...`
function extractObjectIds(content) {
  const ids = [];
  const re = /\*\*OBJECT ID:\*\*\s+`([^`]+)`/g;
  let m;
  while ((m = re.exec(content)) !== null) {
    ids.push(m[1]);
  }
  return ids;
}

// Extract all spacing IDs from content: lines matching #### ↕ `...`
function extractSpacingIds(content) {
  const ids = [];
  const re = /####\s+[↕↔]\s+`([^`]+)`/g;
  let m;
  while ((m = re.exec(content)) !== null) {
    ids.push(m[1]);
  }
  return ids;
}

// Count nav rows: lines starting with '←'
function countNavRows(content) {
  const lines = content.split('\n');
  return lines.filter((l) => l.trim().startsWith('←')).length;
}

// Check Swedish + English content: for each object block, look for SE and EN rows
function checkObjectContent(content) {
  const missing = [];
  // Split into object blocks by #### heading
  const blocks = content.split(/(?=#### )/);
  for (const block of blocks) {
    const idMatch = block.match(/\*\*OBJECT ID:\*\*\s+`([^`]+)`/);
    if (!idMatch) continue;
    const objectId = idMatch[1];
    // Check for SE row
    if (block.includes('| SE |')) {
      const seMatch = block.match(/\| SE \| "([^"]*)"/);
      if (!seMatch || seMatch[1].trim() === '' || seMatch[1].trim() === '—') {
        missing.push(`Object "${objectId}" has empty SE content`);
      }
    } else {
      missing.push(`Object "${objectId}" missing SE content field`);
    }
    // Check for EN row
    if (block.includes('| EN |')) {
      const enMatch = block.match(/\| EN \| "([^"]*)"/);
      if (!enMatch || enMatch[1].trim() === '' || enMatch[1].trim() === '—') {
        missing.push(`Object "${objectId}" has empty EN content`);
      }
    } else {
      missing.push(`Object "${objectId}" missing EN content field`);
    }
  }
  return missing;
}

function validatePage(filePath) {
  const errors = [];
  const warnings = [];

  if (!fs.existsSync(filePath)) {
    return { errors: [`File not found: ${filePath}`], warnings: [], objectCount: 0, spacingCount: 0 };
  }

  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    return { errors: [`Cannot read file: ${error.message}`], warnings: [], objectCount: 0, spacingCount: 0 };
  }

  const pageSlug = path.basename(filePath, '.md');
  // Derive page prefix (strip leading number): "01-start" -> "start"
  const slugParts = pageSlug.split('-');
  const pagePrefix = slugParts.length > 1 ? slugParts.slice(1).join('-') : pageSlug;

  // 1. Required sections
  for (const section of REQUIRED_SECTIONS) {
    if (!content.includes(section)) {
      errors.push(`Missing section: ${section}`);
    }
  }

  // 2. Object IDs — kebab-case and prefix
  const objectIds = extractObjectIds(content);
  const seenIds = new Set();

  for (const id of objectIds) {
    // Kebab-case check
    if (!KEBAB_CASE_RE.test(id)) {
      errors.push(`Object ID not in kebab-case: \`${id}\``);
    }
    // Prefix check
    if (!id.startsWith(pagePrefix + '-')) {
      errors.push(`Object ID missing prefix: \`${id}\` (expected prefix: ${pagePrefix}-)`);
    }
    // Duplicate check
    if (seenIds.has(id)) {
      errors.push(`Duplicate Object ID: \`${id}\``);
    }
    seenIds.add(id);
  }

  // 3. Navigation rows (expect 3)
  const navCount = countNavRows(content);
  if (navCount < 3) {
    errors.push(`Navigation rows: found ${navCount}, expected 3`);
  }

  // 4. Metadata table properties
  for (const prop of REQUIRED_METADATA_PROPS) {
    if (!content.includes(`| ${prop} |`)) {
      errors.push(`Metadata table missing property: ${prop}`);
    }
  }

  // 5. Sketches folder
  const sketchesDir = path.join(path.dirname(filePath), 'sketches');
  if (!fs.existsSync(sketchesDir)) {
    warnings.push('Sketches folder does not exist');
  }

  // 6. Swedish + English content check
  const contentWarnings = checkObjectContent(content);
  for (const w of contentWarnings) {
    warnings.push(w);
  }

  // Spacing IDs
  const spacingIds = extractSpacingIds(content);

  return {
    errors,
    warnings,
    objectCount: objectIds.length,
    spacingCount: spacingIds.length,
  };
}

function formatResult(label, result) {
  const { errors, warnings, objectCount, spacingCount } = result;
  if (errors.length === 0 && warnings.length === 0) {
    return `✓ ${label} — valid (${objectCount} objects, ${spacingCount} spacing objects)\n`;
  }

  const lines = [];
  if (errors.length > 0) {
    lines.push(`✗ ${label} — ${errors.length} error(s):`);
    for (const e of errors) lines.push(`  - ${e}`);
  }
  if (warnings.length > 0) {
    lines.push(`  ${warnings.length} warning(s):`);
    for (const w of warnings) lines.push(`  ! ${w}`);
  }
  return lines.join('\n') + '\n';
}

function getPageFiles(scenarioDir) {
  let entries;
  try {
    entries = fs.readdirSync(scenarioDir, { withFileTypes: true });
  } catch {
    return [];
  }

  return entries
    .filter((e) => e.isDirectory())
    .map((e) => {
      const mdFile = path.join(scenarioDir, e.name, `${e.name}.md`);
      return fs.existsSync(mdFile) ? mdFile : null;
    })
    .filter(Boolean)
    .sort();
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.page && !args.scenario && !args.all) {
    process.stderr.write('Error: --page, --scenario, or --all is required.\n\n');
    printUsage();
    process.exit(1);
  }

  const outputBase = args.output || process.cwd();
  const scenariosBase = path.join(outputBase, 'C-UX-Scenarios');

  let filesToValidate = [];

  if (args.page) {
    filesToValidate = [path.resolve(args.page)];
  } else if (args.scenario) {
    const scenarioSlug = toSlug(args.scenario);
    const scenarioDir = path.join(scenariosBase, scenarioSlug);
    filesToValidate = getPageFiles(scenarioDir);
    if (filesToValidate.length === 0) {
      process.stdout.write(`No pages found in scenario: ${scenarioSlug}\n`);
      process.exit(0);
    }
  } else if (args.all) {
    if (!fs.existsSync(scenariosBase)) {
      process.stderr.write(`Error: C-UX-Scenarios not found at: ${scenariosBase}\n`);
      process.exit(1);
    }
    let entries;
    try {
      entries = fs.readdirSync(scenariosBase, { withFileTypes: true });
    } catch (error) {
      process.stderr.write(`Error reading scenarios: ${error.message}\n`);
      process.exit(1);
    }
    for (const e of entries.filter((e) => e.isDirectory()).sort()) {
      const pages = getPageFiles(path.join(scenariosBase, e.name));
      filesToValidate.push(...pages);
    }
    if (filesToValidate.length === 0) {
      process.stdout.write('No page specs found.\n');
      process.exit(0);
    }
  }

  let hasErrors = false;

  for (const filePath of filesToValidate) {
    const label = path.basename(filePath);
    const result = validatePage(filePath);
    process.stdout.write(formatResult(label, result));
    if (result.errors.length > 0) hasErrors = true;
  }

  process.exit(hasErrors ? 1 : 0);
}

main();
