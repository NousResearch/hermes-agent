#!/bin/sh
set -eu
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64|Linux-x86_64) ;;
  *) echo '{"ok":false,"error":"unsupported_platform","message":"This Skill supports macOS arm64 and Linux x86_64 through scripts/doctor.sh, and Windows x64 through scripts/doctor.ps1."}'; exit 2 ;;
esac
skill_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ -x "$skill_dir/bin/browser-cli" ]; then exec "$skill_dir/bin/browser-cli" doctor; fi
echo '{"ok":false,"error":"command_not_found","message":"Skill-local browser-cli is missing. Run scripts/bootstrap.sh first."}'
exit 1
