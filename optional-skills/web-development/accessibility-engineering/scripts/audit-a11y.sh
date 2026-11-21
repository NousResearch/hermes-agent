#!/usr/bin/env bash
set -e

# accessibility-engineering: audit-a11y.sh
# Runs automated accessibility checks on a given URL or file.
# Outputs structured JSON with violations grouped by severity.
#
# Usage:
#   bash audit-a11y.sh url <url> [--json]
#   bash audit-a11y.sh file <path> [--json]
#
# Requires: npx (@axe-core/cli and pa11y are fetched at the pinned versions below)

SCRIPT_NAME="audit-a11y"
AXE_VERSION="4.13.0"
PA11Y_VERSION="9.1.1"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

MODE="${1:-}"
TARGET="${2:-}"
FORMAT="${3:-text}"

if [ -z "$MODE" ] || [ -z "$TARGET" ]; then
  echo "Usage: bash audit-a11y.sh url <url> [--json]" >&2
  echo "       bash audit-a11y.sh file <path> [--json]" >&2
  exit 1
fi

case "$MODE" in
  url)
    echo "Running axe-core audit on $TARGET..." >&2
    # Hermes: @axe-core/cli bundles a chromedriver that must match the local Chrome major;
    # set AXE_CHROMEDRIVER_PATH to a matching driver when it does not.
    AXE_DRIVER_ARGS=()
    if [ -n "${AXE_CHROMEDRIVER_PATH:-}" ]; then AXE_DRIVER_ARGS=(--chromedriver-path "$AXE_CHROMEDRIVER_PATH"); fi
    if npx --yes "@axe-core/cli@$AXE_VERSION" "$TARGET" "${AXE_DRIVER_ARGS[@]}" --dir "$TEMP_DIR" --save axe-report.json >&2; then
      AXE_EXIT=0
    else
      AXE_EXIT=$?
    fi

    echo "Running pa11y audit on $TARGET..." >&2
    if npx --yes "pa11y@$PA11Y_VERSION" "$TARGET" --reporter json 2>/dev/null > "$TEMP_DIR/pa11y-report.json"; then
      PA11Y_EXIT=0
    else
      PA11Y_EXIT=$?
    fi
    ;;
  file)
    AXE_EXIT=-1
    echo "Running pa11y on file $TARGET..." >&2
    if npx --yes "pa11y@$PA11Y_VERSION" "$TARGET" --reporter json > "$TEMP_DIR/pa11y-report.json" 2>/dev/null; then
      PA11Y_EXIT=0
    else
      PA11Y_EXIT=$?
    fi
    echo "File-based axe analysis limited; prefer URL mode for full coverage." >&2
    ;;
  *)
    echo "Unknown mode: $MODE. Use 'url' or 'file'." >&2
    exit 1
    ;;
esac

report_status() {
  local tool="$1"
  local report="$2"
  local exit_code="$3"
  if [ ! -s "$report" ]; then
    printf 'failed'
    return
  fi

  # axe runs without --exit, so any non-zero status is an execution failure.
  # pa11y uses exit 2 when a valid report contains violations; other non-zero
  # statuses mean the audit itself failed.
  if { [ "$tool" = "axe" ] && [ "$exit_code" -ne 0 ]; } || \
     { [ "$tool" = "pa11y" ] && [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne 2 ]; }; then
    printf 'failed'
  elif node -e "
    const fs = require('fs');
    const tool = process.argv[1];
    const report = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    const valid = tool === 'axe'
      ? Array.isArray(report) && report.length > 0 && report.every(run => run && Array.isArray(run.violations))
      : Array.isArray(report);
    if (!valid) process.exit(1);
  " "$tool" "$report" >/dev/null 2>&1; then
    printf 'completed'
  else
    printf 'invalid-report'
  fi
}

if [ "$MODE" = "url" ]; then
  AXE_STATUS=$(report_status axe "$TEMP_DIR/axe-report.json" "$AXE_EXIT")
else
  AXE_STATUS="not-run"
fi
PA11Y_STATUS=$(report_status pa11y "$TEMP_DIR/pa11y-report.json" "$PA11Y_EXIT")

# A URL audit promises both engines. A valid report is the success signal because
# pa11y may return non-zero when it found violations. Never turn a partial run
# into a clean result just because the remaining engine returned an empty list.
if [ "$PA11Y_STATUS" != "completed" ] || { [ "$MODE" = "url" ] && [ "$AXE_STATUS" != "completed" ]; }; then
  echo "$SCRIPT_NAME: incomplete audit; refusing to report a violation total" >&2
  echo "$SCRIPT_NAME: tool status: @axe-core/cli@$AXE_VERSION=$AXE_STATUS (exit $AXE_EXIT), pa11y@$PA11Y_VERSION=$PA11Y_STATUS (exit $PA11Y_EXIT)" >&2
  exit 1
fi

# Merge results into structured JSON output
RESULT_FILE="$TEMP_DIR/result.json"

cat > "$RESULT_FILE" << RESULT_EOF
{
  "tool": "audit-a11y",
  "tools": [
    {"name": "@axe-core/cli", "version": "$AXE_VERSION", "status": "$AXE_STATUS", "exitCode": $AXE_EXIT},
    {"name": "pa11y", "version": "$PA11Y_VERSION", "status": "$PA11Y_STATUS", "exitCode": $PA11Y_EXIT}
  ],
  "summary": {
    "violations": 0,
    "critical": 0,
    "serious": 0,
    "moderate": 0,
    "minor": 0
  },
  "violations": [],
  "recommendations": []
}
RESULT_EOF

# Parse axe results
if [ -f "$TEMP_DIR/axe-report.json" ]; then
  node -e "
    const fs = require('fs');
    const result = JSON.parse(fs.readFileSync('$TEMP_DIR/result.json', 'utf8'));
    try {
      const axe = JSON.parse(fs.readFileSync('$TEMP_DIR/axe-report.json', 'utf8'));
      // @axe-core/cli saves an array with one result per URL, and no results wrapper.
      for (const run of (Array.isArray(axe) ? axe : [axe])) {
        const violations = run.violations || [];
        result.summary.violations += violations.length;
        for (const v of violations) {
          const sev = v.impact || 'minor';
          if (result.summary[sev] !== undefined) result.summary[sev]++;
          result.violations.push({
            source: 'axe-core',
            id: v.id,
            impact: sev,
            description: v.description,
            help: v.help,
            helpUrl: v.helpUrl,
            tags: v.tags || [],
            nodes: (v.nodes || []).map(n => ({
              html: n.html,
              target: n.target,
              failureSummary: n.failureSummary
            }))
          });
        }
      }
    } catch (e) {
      // An empty report is covered by the guard above; a non-empty one that does
      // not parse is a tool failure, not a clean result.
      if (fs.statSync('$TEMP_DIR/axe-report.json').size > 0) {
        console.error('$SCRIPT_NAME: axe-report.json is not valid JSON: ' + e.message);
        process.exit(1);
      }
    }
    fs.writeFileSync('$TEMP_DIR/result.json', JSON.stringify(result, null, 2));
  "
fi

# Parse pa11y results
if [ -f "$TEMP_DIR/pa11y-report.json" ]; then
  node -e "
    const fs = require('fs');
    const result = JSON.parse(fs.readFileSync('$TEMP_DIR/result.json', 'utf8'));
    try {
      const pa11y = JSON.parse(fs.readFileSync('$TEMP_DIR/pa11y-report.json', 'utf8'));
      const issues = Array.isArray(pa11y) ? pa11y : (pa11y.issues || []);
      result.summary.violations += issues.length;
      for (const issue of issues) {
        // pa11y has no impact scale: typeCode 1 is "error", and by default pa11y
        // reports only errors. Mapping that to critical would turn every pa11y
        // result into a /ship NO-GO, because axe rates the same failure serious.
        const sev = (issue.typeCode === 1 ? 'serious' : issue.typeCode === 2 ? 'moderate' : 'minor');
        if (result.summary[sev] !== undefined) result.summary[sev]++;
        result.violations.push({
          source: 'pa11y',
          code: issue.code,
          impact: sev,
          message: issue.message,
          selector: issue.selector,
          context: issue.context,
          type: issue.type
        });
      }
    } catch (e) {
      // Same as the axe block: an empty report is covered by the guard above.
      if (fs.statSync('$TEMP_DIR/pa11y-report.json').size > 0) {
        console.error('$SCRIPT_NAME: pa11y-report.json is not valid JSON: ' + e.message);
        process.exit(1);
      }
    }
    fs.writeFileSync('$TEMP_DIR/result.json', JSON.stringify(result, null, 2));
  "
fi

if [ "$FORMAT" = "--json" ]; then
  cat "$RESULT_FILE"
else
  node -e "
    const r = JSON.parse(require('fs').readFileSync('$RESULT_FILE', 'utf8'));
    console.log('Accessibility Audit Results');
    console.log('===========================');
    console.log('Tools:');
    for (const tool of r.tools) {
      console.log('  ' + tool.name + '@' + tool.version + ': ' + tool.status + ' (exit ' + tool.exitCode + ')');
    }
    console.log('');
    console.log('Total violations: ' + r.summary.violations);
    console.log('  Critical: ' + r.summary.critical);
    console.log('  Serious:  ' + r.summary.serious);
    console.log('  Moderate: ' + r.summary.moderate);
    console.log('  Minor:    ' + r.summary.minor);
    console.log('');
    for (const v of r.violations) {
      console.log('[' + v.impact.toUpperCase() + '] ' + (v.help || v.message || v.id));
      if (v.nodes && v.nodes[0]) console.log('  → ' + v.nodes[0].target);
      if (v.selector) console.log('  → ' + v.selector);
      console.log('');
    }
  "
fi
