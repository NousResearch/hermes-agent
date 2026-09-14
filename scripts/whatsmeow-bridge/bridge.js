#!/usr/bin/env node
const { spawnSync } = require('child_process');
const path = require('path');
const bridge = path.join(__dirname, 'whatsmeow-bridge');
const result = spawnSync(bridge, process.argv.slice(2), { stdio: 'inherit' });
process.exit(result.status || 0);
