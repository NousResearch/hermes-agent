// wds-init-page.js — WDS scaffold: initialize new page spec
// Usage: node src/scripts/wds-init-page.js --page "01 Start" --scenario "01 Onboarding"

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
      'Usage: node src/scripts/wds-init-page.js --page "01 Start" --scenario "01 Onboarding" [options]',
      '',
      'Required:',
      '  --page        Page name with number, e.g. "01 Start"',
      '  --scenario    Scenario name, e.g. "01 New User Onboarding"',
      '',
      'Optional:',
      '  --platform    Platform value (default: "Mobile web")',
      '  --visibility  Visibility value (default: "Public")',
      '  --output      Base path to write to (default: current directory)',
      '',
    ].join('\n'),
  );
}

function buildTemplate({ pageSlug, pageName, scenarioSlug, scenarioName, platform, visibility }) {
  const sketchFile = `sketches/${pageSlug}-concept.jpg`;

  const navEmpty = `← [Previous]() | [Next →]()`;

  const metaTable = [
    '| Property | Value |',
    '|----------|-------|',
    `| Scenario | ${scenarioName} |`,
    `| Page Number | ${pageName.split(' ')[0]} |`,
    `| Platform | ${platform} |`,
    `| Page Type | — |`,
    `| Viewport | — |`,
    `| Interaction | — |`,
    `| Visibility | ${visibility} |`,
  ].join('\n');

  const overviewSection = [
    '| Property | Value |',
    '|----------|-------|',
    '| Purpose | — |',
    '| User Situation | — |',
    '| Success Criteria | — |',
    '| Entry Points | — |',
    '| Exit Points | — |',
  ].join('\n');

  const spacingTable = [
    '| Token | Direction | Type | Size | Reason |',
    '|-------|-----------|------|------|--------|',
    `| \`${pageSlug}-v-space-md\` | Vertical | space | md | — |`,
  ].join('\n');

  const typographyTable = [
    '| Element | Semantic | Size | Weight | Typeface |',
    '|---------|----------|------|--------|----------|',
    '| Page title | H1 | — | — | — |',
  ].join('\n');

  const statesTable = [
    '| State | When | Appearance | Actions |',
    '|-------|------|------------|---------|',
    '| Default | On load | — | — |',
  ].join('\n');

  return [
    navEmpty,
    '',
    `![${pageName}](${sketchFile})`,
    '',
    navEmpty,
    '',
    '---',
    '',
    `# ${pageName}`,
    '',
    '## Page Metadata',
    '',
    metaTable,
    '',
    '---',
    '',
    '## Overview',
    '',
    overviewSection,
    '',
    '---',
    '',
    '## Reference Materials',
    '',
    '- —',
    '',
    '---',
    '',
    '## Layout Structure',
    '',
    '```',
    '+---------------------------+',
    '|         Header            |',
    '+---------------------------+',
    '|                           |',
    '|         Content           |',
    '|                           |',
    '+---------------------------+',
    '|         Footer            |',
    '+---------------------------+',
    '```',
    '',
    '---',
    '',
    '## Spacing',
    '',
    spacingTable,
    '',
    '---',
    '',
    '## Typography',
    '',
    typographyTable,
    '',
    '---',
    '',
    '## Page Sections',
    '',
    `### Section: Main`,
    '',
    `<!-- Objects go here. Use wds-add-object.js to append. -->`,
    '',
    '---',
    '',
    '## Page States',
    '',
    statesTable,
    '',
    '---',
    '',
    '## Open Questions',
    '',
    '- —',
    '',
    '---',
    '',
    '## Checklist',
    '',
    '- [ ] Navigation links updated',
    '- [ ] Sketch added',
    '- [ ] All objects have SE + EN content',
    '- [ ] All Object IDs use correct prefix',
    '- [ ] Page States defined',
    '- [ ] Spacing tokens defined',
    '',
    navEmpty,
    '',
  ].join('\n');
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.page || !args.scenario) {
    process.stderr.write('Error: --page and --scenario are required.\n\n');
    printUsage();
    process.exit(1);
  }

  const pageName = args.page;
  const scenarioName = args.scenario;
  const platform = args.platform || 'Mobile web';
  const visibility = args.visibility || 'Public';
  const outputBase = args.output || process.cwd();

  const pageSlug = toSlug(pageName);
  const scenarioSlug = toSlug(scenarioName);

  const pageDir = path.join(outputBase, 'C-UX-Scenarios', scenarioSlug, pageSlug);
  const sketchesDir = path.join(pageDir, 'sketches');
  const pageFile = path.join(pageDir, `${pageSlug}.md`);

  try {
    fs.mkdirSync(pageDir, { recursive: true });
    fs.mkdirSync(sketchesDir, { recursive: true });
  } catch (error) {
    process.stderr.write(`Error creating directories: ${error.message}\n`);
    process.exit(1);
  }

  const content = buildTemplate({ pageSlug, pageName, scenarioSlug, scenarioName, platform, visibility });

  try {
    fs.writeFileSync(pageFile, content, 'utf8');
  } catch (error) {
    process.stderr.write(`Error writing file: ${error.message}\n`);
    process.exit(1);
  }

  process.stdout.write(`✓ Created ${pageSlug}.md\n`);
  process.stdout.write(`  Path: ${pageFile}\n`);
  process.stdout.write(`  Run wds-nav.js to update navigation links.\n`);
}

main();
