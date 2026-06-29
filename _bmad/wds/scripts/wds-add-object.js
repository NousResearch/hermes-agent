// wds-add-object.js — WDS scaffold: append an object spec block to a page spec file
// Usage: node src/scripts/wds-add-object.js --page "C-UX-Scenarios/01-onboarding/01-start/01-start.md" \
//          --section "Hero" --object "Primary Headline" --component "H1 heading" \
//          --se "Välkommen" --en "Welcome"

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
      'Usage: node src/scripts/wds-add-object.js --page <path> --section <name> --object <name> [options]',
      '',
      'Required:',
      '  --page        Path to the page spec .md file',
      '  --section     Section name (e.g. "Hero")',
      '  --object      Object name (e.g. "Primary Headline")',
      '',
      'Optional:',
      '  --component   Component name (default: "—")',
      '  --translation Translation key (auto-derived if omitted)',
      '  --se          Swedish text content',
      '  --en          English text content',
      '  --behavior    Behavior description (e.g. "onClick: submit form")',
      '  --component-path  Path to component file (default: "—")',
      '',
    ].join('\n'),
  );
}

// Derive page slug from file path: "01-start/01-start.md" -> "01-start"
function pageSlugFromPath(filePath) {
  const base = path.basename(filePath, '.md');
  return base;
}

// Derive Object ID: pageSlug + sectionSlug + objectSlug
// e.g. page=01-start, section=Hero, object=Primary Headline -> 01-start-hero-primary-headline
function deriveObjectId(pageSlug, sectionName, objectName) {
  // Strip leading page number from pageSlug for ID prefix
  // "01-start" -> "start", "02-signup" -> "signup"
  const slugParts = pageSlug.split('-');
  const pagePrefix = slugParts.length > 1 ? slugParts.slice(1).join('-') : pageSlug;
  const sectionSlug = toSlug(sectionName);
  const objectSlug = toSlug(objectName);
  return `${pagePrefix}-${sectionSlug}-${objectSlug}`;
}

function buildObjectBlock({ objectName, objectId, component, componentPath, translationKey, se, en, behavior }) {
  const compDisplay = componentPath && componentPath !== '—' ? `[${component}](${componentPath})` : component || '—';

  const lines = [
    `#### ${objectName}`,
    '',
    `**OBJECT ID:** \`${objectId}\``,
    '',
    '| Property | Value |',
    '|----------|-------|',
    `| Component | ${compDisplay} |`,
    `| Translation Key | \`${translationKey}\` |`,
    `| SE | "${se || '—'}" |`,
    `| EN | "${en || '—'}" |`,
    `| Behavior | ${behavior || '—'} |`,
    '',
  ];

  return lines.join('\n');
}

// Insert content after a section heading. Creates the section heading if it doesn't exist.
function insertUnderSection(content, sectionHeading, objectBlock) {
  const lines = content.split('\n');
  const headingLine = `### Section: ${sectionHeading}`;
  const headingIdx = lines.findIndex((l) => l.trim() === headingLine);

  if (headingIdx === -1) {
    // Section doesn't exist — append it before the next ## heading after ## Page Sections
    const pageSectionsIdx = lines.findIndex((l) => l.trim() === '## Page Sections');
    if (pageSectionsIdx === -1) {
      // Just append at end before last nav row
      return content + `\n${headingLine}\n\n${objectBlock}\n`;
    }

    // Find end of ## Page Sections block
    let insertIdx = pageSectionsIdx + 1;
    while (insertIdx < lines.length) {
      const t = lines[insertIdx].trim();
      if (t.startsWith('## ') || t === '---') break;
      insertIdx++;
    }

    const before = lines.slice(0, insertIdx);
    const after = lines.slice(insertIdx);
    return [...before, '', headingLine, '', objectBlock, ...after].join('\n');
  } else {
    // Find the end of this section (next ### or ## or end of file)
    let insertIdx = headingIdx + 1;
    // Skip blank lines after heading
    while (insertIdx < lines.length && lines[insertIdx].trim() === '') insertIdx++;
    // Skip comment lines
    while (insertIdx < lines.length && lines[insertIdx].trim().startsWith('<!--')) insertIdx++;

    // Find end of section
    let endIdx = insertIdx;
    while (endIdx < lines.length) {
      const t = lines[endIdx].trim();
      if (t.startsWith('### ') || t.startsWith('## ') || t === '---') break;
      endIdx++;
    }

    // Insert object block before end of section
    const before = lines.slice(0, endIdx);
    const after = lines.slice(endIdx);
    return [...before, '', objectBlock, ...after].join('\n');
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.page || !args.section || !args.object) {
    process.stderr.write('Error: --page, --section, and --object are required.\n\n');
    printUsage();
    process.exit(1);
  }

  const filePath = path.resolve(args.page);

  if (!fs.existsSync(filePath)) {
    process.stderr.write(`Error: File not found: ${filePath}\n`);
    process.exit(1);
  }

  const pageSlug = pageSlugFromPath(filePath);
  const objectId = deriveObjectId(pageSlug, args.section, args.object);

  // Auto-derive translation key from objectId
  const translationKey = args.translation || objectId.replaceAll('-', '.');

  const objectBlock = buildObjectBlock({
    objectName: args.object,
    objectId,
    component: args.component || '—',
    componentPath: args['component-path'] || '—',
    translationKey,
    se: args.se || '',
    en: args.en || '',
    behavior: args.behavior || '—',
  });

  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    process.stderr.write(`Error reading file: ${error.message}\n`);
    process.exit(1);
  }

  // Check for duplicate object ID
  if (content.includes(`\`${objectId}\``)) {
    process.stderr.write(`Error: Object ID already exists in file: ${objectId}\n`);
    process.exit(1);
  }

  const updated = insertUnderSection(content, args.section, objectBlock);

  try {
    fs.writeFileSync(filePath, updated, 'utf8');
  } catch (error) {
    process.stderr.write(`Error writing file: ${error.message}\n`);
    process.exit(1);
  }

  process.stdout.write(`✓ Added object ${objectId}\n`);
  process.stdout.write(`  File: ${filePath}\n`);
}

main();
