const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const missing = [];
for (const container of pkg.contributes.viewsContainers.activitybar || []) {
  if (container.icon && !fs.existsSync(path.join(root, container.icon))) missing.push(container.icon);
}
if (!fs.existsSync(path.join(root, pkg.main))) missing.push(pkg.main);
if (missing.length) {
  console.error(`Missing package files: ${missing.join(', ')}`);
  process.exit(1);
}
console.log('package-check ok');
