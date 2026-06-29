// wds-init-scenario.js — WDS scaffold: initialize new scenario folder
// Usage: node src/scripts/wds-init-scenario.js --scenario "01 Onboarding" --description "New user first visit to account creation"

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
      'Usage: node src/scripts/wds-init-scenario.js --scenario "01 Onboarding" [options]',
      '',
      'Required:',
      '  --scenario      Scenario name with number, e.g. "01 New User Onboarding"',
      '',
      'Optional:',
      '  --description   Short description of the scenario',
      '  --output        Base path to write to (default: current directory)',
      '',
    ].join('\n'),
  );
}

function buildReadme({ scenarioName, scenarioSlug, description }) {
  const scenarioNumber = scenarioName.split(' ')[0];
  const desc = description || '—';

  return [
    `# Scenario ${scenarioNumber}: ${scenarioName}`,
    '',
    `**Description:** ${desc}`,
    '',
    '**Trigger Map:** [Link to trigger map]()',
    '',
    '---',
    '',
    '## Pages',
    '',
    '| # | Page | File | Status |',
    '|---|------|------|--------|',
    '| — | (no pages yet) | — | — |',
    '',
    '---',
    '',
    '## Notes',
    '',
    '- Add pages with: `node src/scripts/wds-init-page.js --scenario "' + scenarioName + '" --page "01 Start"`',
    '- Update navigation after adding pages: `node src/scripts/wds-nav.js --scenario "' + scenarioName + '"`',
    '- Validate pages: `node src/scripts/wds-validate.js --scenario "' + scenarioName + '"`',
    '',
  ].join('\n');
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.scenario) {
    process.stderr.write('Error: --scenario is required.\n\n');
    printUsage();
    process.exit(1);
  }

  const scenarioName = args.scenario;
  const description = args.description || '';
  const outputBase = args.output || process.cwd();

  const scenarioSlug = toSlug(scenarioName);
  const scenarioDir = path.join(outputBase, 'C-UX-Scenarios', scenarioSlug);
  const readmeFile = path.join(scenarioDir, 'README.md');

  if (fs.existsSync(scenarioDir)) {
    process.stderr.write(`Error: Scenario folder already exists: ${scenarioDir}\n`);
    process.exit(1);
  }

  try {
    fs.mkdirSync(scenarioDir, { recursive: true });
  } catch (error) {
    process.stderr.write(`Error creating directory: ${error.message}\n`);
    process.exit(1);
  }

  const content = buildReadme({ scenarioName, scenarioSlug, description });

  try {
    fs.writeFileSync(readmeFile, content, 'utf8');
  } catch (error) {
    process.stderr.write(`Error writing README: ${error.message}\n`);
    process.exit(1);
  }

  process.stdout.write(`✓ Created scenario ${scenarioSlug}/\n`);
  process.stdout.write(`  Path: ${scenarioDir}\n`);
  process.stdout.write(`  README: ${readmeFile}\n`);
}

main();
