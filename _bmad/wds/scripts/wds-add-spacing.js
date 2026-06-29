// wds-add-spacing.js — WDS scaffold: append a spacing object to a page spec file
// Usage: node src/scripts/wds-add-spacing.js --page "C-UX-Scenarios/01-onboarding/01-start/01-start.md" \
//          --direction v --type space --size xl --reason "major section boundary between hero and features"

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

function printUsage() {
  process.stdout.write(
    [
      'Usage: node src/scripts/wds-add-spacing.js --page <path> --direction <v|h> --type <type> --size <size> [options]',
      '',
      'Required:',
      '  --page        Path to the page spec .md file',
      '  --direction   v (vertical) or h (horizontal)',
      '  --type        space | separator | line',
      '  --size        zero | sm | md | lg | xl | 2xl | 3xl | flex',
      '',
      'Optional:',
      '  --reason      Why this spacing exists',
      '',
      'Valid directions: v, h',
      'Valid types:      space, separator, line',
      'Valid sizes:      zero, sm, md, lg, xl, 2xl, 3xl, flex',
      '',
    ].join('\n'),
  );
}

const VALID_DIRECTIONS = ['v', 'h'];
const VALID_TYPES = ['space', 'separator', 'line'];
const VALID_SIZES = ['zero', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', 'flex'];

// Derive page prefix from slug: "01-start" -> "start"
function pagePrefix(slug) {
  const parts = slug.split('-');
  return parts.length > 1 ? parts.slice(1).join('-') : slug;
}

function pageSlugFromPath(filePath) {
  return path.basename(filePath, '.md');
}

function buildSpacingBlock(spacingId, reason) {
  const icon = '↕';
  const reasonText = reason ? ` — ${reason}` : '';
  return `#### ${icon} \`${spacingId}\`${reasonText}\n`;
}

function appendToSpacingSection(content, spacingBlock) {
  const lines = content.split('\n');
  const spacingIdx = lines.findIndex((l) => l.trim() === '## Spacing');

  if (spacingIdx === -1) {
    // No spacing section — append before first ## after metadata
    return content + `\n## Spacing\n\n${spacingBlock}\n`;
  }

  // Find end of spacing section (next ## or ---)
  let endIdx = spacingIdx + 1;
  while (endIdx < lines.length) {
    const t = lines[endIdx].trim();
    if ((t.startsWith('## ') && t !== '## Spacing') || t === '---') break;
    endIdx++;
  }

  // Insert just before the end marker
  const before = lines.slice(0, endIdx);
  const after = lines.slice(endIdx);
  return [...before, spacingBlock, ...after].join('\n');
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    process.exit(0);
  }

  if (!args.page || !args.direction || !args.type || args.size === 0) {
    process.stderr.write('Error: --page, --direction, --type, and --size are required.\n\n');
    printUsage();
    process.exit(1);
  }

  if (!VALID_DIRECTIONS.includes(args.direction)) {
    process.stderr.write(`Error: Invalid direction "${args.direction}". Must be: ${VALID_DIRECTIONS.join(', ')}\n`);
    process.exit(1);
  }

  if (!VALID_TYPES.includes(args.type)) {
    process.stderr.write(`Error: Invalid type "${args.type}". Must be: ${VALID_TYPES.join(', ')}\n`);
    process.exit(1);
  }

  if (!VALID_SIZES.includes(args.size)) {
    process.stderr.write(`Error: Invalid size "${args.size}". Must be: ${VALID_SIZES.join(', ')}\n`);
    process.exit(1);
  }

  const filePath = path.resolve(args.page);

  if (!fs.existsSync(filePath)) {
    process.stderr.write(`Error: File not found: ${filePath}\n`);
    process.exit(1);
  }

  const slug = pageSlugFromPath(filePath);
  const prefix = pagePrefix(slug);
  const spacingId = `${prefix}-${args.direction}-${args.type}-${args.size}`;
  const reason = args.reason || '';

  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    process.stderr.write(`Error reading file: ${error.message}\n`);
    process.exit(1);
  }

  // Check for duplicate spacing ID
  if (content.includes(`\`${spacingId}\``)) {
    process.stderr.write(`Error: Spacing ID already exists in file: ${spacingId}\n`);
    process.stderr.write('Use a different combination of direction/type/size or manually edit the file.\n');
    process.exit(1);
  }

  const spacingBlock = buildSpacingBlock(spacingId, reason);
  const updated = appendToSpacingSection(content, spacingBlock);

  try {
    fs.writeFileSync(filePath, updated, 'utf8');
  } catch (error) {
    process.stderr.write(`Error writing file: ${error.message}\n`);
    process.exit(1);
  }

  process.stdout.write(`✓ Added spacing object ${spacingId}\n`);
  process.stdout.write(`  File: ${filePath}\n`);
}

main();
