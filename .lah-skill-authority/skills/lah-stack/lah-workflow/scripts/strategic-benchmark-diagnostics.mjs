#!/usr/bin/env node
/**
 * Strategic Benchmark Diagnostics
 *
 * Run the provider-backed 9-question benchmark and evaluate results
 * through the V3 strategic certification gate in one shot.
 *
 * Usage:
 *   node scripts/strategic-benchmark-diagnostics.mjs       # full run + evaluate
 *   node scripts/strategic-benchmark-diagnostics.mjs --dry  # evaluate existing report only
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');
const REPORT_PATH = resolve(REPO_ROOT, 'test/reports/cloe-operator-benchmark-latest.json');
const GATE_MODULE_PATH = resolve(REPO_ROOT, 'src/cognitive/cloe-v3-strategic-benchmark-certification.js');

const args = process.argv.slice(2);
const dryRun = args.includes('--dry');

async function main() {
  console.log('=== Strategic Benchmark Diagnostics ===\n');

  if (!dryRun) {
    console.log('Phase 1: Running provider-backed benchmark (9 questions)...');
    console.log('(This will take 2-4 minutes)\n');

    try {
      const result = execFileSync('npm', ['run', 'benchmark:cloe-operator'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 300000,
      });

      // Extract JSON summary from output
      const lines = result.split('\n');
      let jsonStr = '', braceCount = 0, inJson = false;
      for (const ch of result) {
        if (ch === '{') { inJson = true; braceCount++; jsonStr += ch; }
        else if (ch === '}') { braceCount--; jsonStr += ch; if (braceCount === 0) break; }
        else if (inJson) jsonStr += ch;
      }

      if (jsonStr) {
        const summary = JSON.parse(jsonStr);
        console.log('Benchmark complete:');
        console.log('  useful:', summary.useful, 'partial:', summary.partial,
          'failed:', summary.failed, 'fallback:', summary.fallback);
        console.log('  avg score:', summary.average_score, 'validated:', summary.answer_quality_validated);
        console.log('  regression guard:', summary.regression_guard?.pass ? 'PASS' : 'FAIL');
        console.log();
      }
    } catch (e) {
      console.log('BENCHMARK ERROR:', e.message.slice(0, 300));
      process.exit(1);
    }
  } else {
    console.log('Phase 1: Skipped (--dry mode)\n');
  }

  // Phase 2: Evaluate with V3 strategic gate
  console.log('Phase 2: V3 Strategic Gate evaluation...\n');

  if (!existsSync(GATE_MODULE_PATH)) {
    console.log('V3 strategic gate module not found at:', GATE_MODULE_PATH);
    console.log('Skipping Phase 2.');
    process.exit(0);
  }

  if (!existsSync(REPORT_PATH)) {
    console.log('Benchmark report not found at:', REPORT_PATH);
    console.log('Run the benchmark first (omit --dry).');
    process.exit(1);
  }

  try {
    const gate = await import(GATE_MODULE_PATH);
    const result = gate.evaluateV3StrategicBenchmark(true);

    console.log('Technical certification:', result.technical_certification);
    console.log('Strategic certification:', result.strategic_certification);
    console.log('Weighted strategic score:', result.weighted_strategic_score + '%');
    console.log('Pass:', result.strategic_pass, 'Partial:', result.strategic_partial,
      'Fail:', result.strategic_fail);
    console.log();

    console.log('Per-question:');
    for (const q of result.question_results) {
      const stale = q.stale_context_found ? ' [STALE]' : '';
      console.log('  ' + q.id + ': ' + q.existing_score + ' -> V3=' + q.v3_strategic_score +
        ' (' + q.v3_strategic_status + ')' + stale);
    }
    console.log();

    console.log('Stale context: ' + result.stale_context.total_issues + ' issues, ' +
      result.stale_context.high_risk_questions + ' high-risk questions');
    console.log();

    console.log('Criteria:');
    for (const [id, c] of Object.entries(result.overall_criteria)) {
      console.log('  ' + c.name + ': ' + c.average_score + '%');
    }

    console.log('\n=== Verdict: ' + result.strategic_certification + ' ===');
  } catch (e) {
    console.log('GATE EVALUATION ERROR:', e.message.slice(0, 300));
    process.exit(1);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
