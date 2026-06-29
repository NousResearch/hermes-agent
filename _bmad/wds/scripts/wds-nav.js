// wds-nav.js — WDS scaffold: generate/update navigation links across all pages in a scenario
// Usage: node src/scripts/wds-nav.js --scenario "01 Onboarding"
//        node src/scripts/wds-nav.js --all

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
      'Usage: node src/scripts/wds-nav.js --scenario "01 Onboarding"',
      '       node src/scripts/wds-nav.js --all',
      '',
      'Options:',
      '  --scenario    Scenario name or slug to update',
      '  --all         Update all scenarios',
      '  --output      Base path (default: current directory)',
      '',
    ].join('\n'),
  );
}

// Build a human-readable page name from the slug for nav labels
// e.g. "01-start" -> "01 Start"
function slugToLabel(slug) {
  return slug
    .split('-')
    .map((part, i) => (i === 0 ? part : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(' ');
}

function buildNavRow(prev, next) {
  const leftPart = prev ? `← [${slugToLabel(prev.slug)}](../${prev.slug}/${prev.slug}.md)` : '←';
  const rightPart = next ? `[${slugToLabel(next.slug)} →](../${next.slug}/${next.slug}.md)` : '→';
  return `${leftPart} | ${rightPart}`;
}

// Replace all 3 occurrences of navigation rows in a page file.
// Nav rows are lines matching the pattern: starts with '←' or contains '| [' and ends with '→'
// We identify them by a simple pattern: line that starts with "←" (after trim).
function updateNavInContent(content, navRow) {
  const lines = content.split('\n');
  const result = [];
  let navCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    // Match navigation rows: lines that start with "←" (our nav format)
    if (trimmed.startsWith('←')) {
      result.push(navRow);
      navCount++;
    } else {
      result.push(line);
    }
  }

  return { content: result.join('\n'), navCount };
}

function getPageFolders(scenarioDir) {
  let entries;
  try {
    entries = fs.readdirSync(scenarioDir, { withFileTypes: true });
  } catch {
    return [];
  }

  return entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .filter((name) => {
      // Must have a matching .md file inside
      const mdFile = path.join(scenarioDir, name, `${name}.md`);
      return fs.existsSync(mdFile);
    })
    .sort(); // Sort alphabetically — page numbers ensure correct order
}

function processScenario(scenariosBase, scenarioSlug) {
  const scenarioDir = path.join(scenariosBase, scenarioSlug);

  if (!fs.existsSync(scenarioDir)) {
    process.stderr.write(`Error: Scenario not found: ${scenarioDir}\n`);
    return false;
  }

  const pageSlugs = getPageFolders(scenarioDir);

  if (pageSlugs.length === 0) {
    process.stdout.write(`  ${scenarioSlug}: no pages found, skipping.\n`);
    return true;
  }

  let updated = 0;

  for (let i = 0; i < pageSlugs.length; i++) {
    const slug = pageSlugs[i];
    const prev = i > 0 ? { slug: pageSlugs[i - 1] } : null;
    const next = i < pageSlugs.length - 1 ? { slug: pageSlugs[i + 1] } : null;
    const navRow = buildNavRow(prev, next);

    const mdFile = path.join(scenarioDir, slug, `${slug}.md`);
    let content;
    try {
      content = fs.readFileSync(mdFile, 'utf8');
    } catch (error) {
      process.stderr.write(`  Error reading ${mdFile}: ${error.message}\n`);
      continue;
    }

    const { content: newContent, navCount } = updateNavInContent(content, navRow);

    if (navCount === 0) {
      process.stderr.write(`  Warning: No navigation rows found in ${slug}.md\n`);
    }

    try {
      fs.writeFileSync(mdFile, newContent, 'utf8');
      updated++;
    } catch (error) {
      process.stderr.write(`  Error writing ${mdFile}: ${error.message}\n`);
    }
  }

  process.stdout.write(`✓ Updated navigation for ${updated} pages in ${scenarioSlug}\n`);
  return true;
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.scenario && !args.all) {
    process.stderr.write('Error: --scenario or --all is required.\n\n');
    printUsage();
    process.exit(1);
  }

  const outputBase = args.output || process.cwd();
  const scenariosBase = path.join(outputBase, 'C-UX-Scenarios');

  if (!fs.existsSync(scenariosBase)) {
    process.stderr.write(`Error: C-UX-Scenarios directory not found at: ${scenariosBase}\n`);
    process.exit(1);
  }

  if (args.all) {
    let entries;
    try {
      entries = fs.readdirSync(scenariosBase, { withFileTypes: true });
    } catch (error) {
      process.stderr.write(`Error reading C-UX-Scenarios: ${error.message}\n`);
      process.exit(1);
    }

    const scenarios = entries
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();

    if (scenarios.length === 0) {
      process.stdout.write('No scenarios found.\n');
      process.exit(0);
    }

    let allOk = true;
    for (const slug of scenarios) {
      if (!processScenario(scenariosBase, slug)) allOk = false;
    }
    process.exit(allOk ? 0 : 1);
  } else {
    const scenarioSlug = toSlug(args.scenario);
    const ok = processScenario(scenariosBase, scenarioSlug);
    process.exit(ok ? 0 : 1);
  }
}

main();
