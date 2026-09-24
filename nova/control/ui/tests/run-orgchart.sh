#!/bin/sh
# Team chart layout tests. From nova/control/ui, after `npm ci`:
#   sh tests/run-orgchart.sh
set -e
npx tsc src/lib/orgchart.ts --outDir tests/.build --module es2020 --target es2020 \
  --strict --moduleResolution node --skipLibCheck
node tests/orgchart.test.mjs
rm -rf tests/.build
