# Operator Testing Gate — Gate 9.5

Post-merge, pre-memory-lock validation against real environments. This gate sits between **Gate 9 (PR & Merge)** and **Gate 10 (Memory Lock)**.

## Why a separate gate

Unit tests (Gate 6) verify code correctness in isolation. Operator tests verify the deployed system actually works — endpoints respond, integrations resolve, performance holds, no regressions appear in production-like conditions.

This gate is **optional but recommended** for any mission that touches live-adjacent infrastructure (WordPress, gateway, provider integration, API endpoints). Skip only for pure documentation or memory-only missions.

## Gate-pass

All smoke checks pass against the deployed/staging environment. Zero HTTP 5xx, zero timeouts on health endpoints, critical user journeys return expected responses. Report documents what was tested, what passed, and any anomalies observed.

## Tool choice by complexity

| Complexity | Tool | Why |
|------------|------|-----|
| **Minimal** (1-5 endpoints) | `curl` + bash script | Zero deps, runs in CI, exits non-zero on failure |
| **Light** (API-only checks) | `asm89/smoke.sh` | Shell framework, ~100 lines, define checks declaratively |
| **Medium** (browser flows) | `basecamp/upright` | Playwright-based synthetic monitoring with Prometheus metrics |
| **Medium** (monitoring-as-code) | `checkly/checkly-cli` | YAML checks committed in repo, Playwright native |
| **Heavy** (load + validation) | `grafana/k6` | JS threshold-based tests, standalone binary, CI-native |
| **Heavy** (traffic replay) | `keploy/keploy` | Record production traffic, replay against new version |

## Recommended patterns for LAH Stack

### Quick smoke (bash, zero deps)

```bash
#!/bin/bash
# operator-smoke.sh — run after merge, before memory lock
BASE_URL="${1:-https://liveaccesshub.com}"
ERRORS=0

check() {
  local url="$1"
  local expected="$2"
  local desc="$3"
  local status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url")
  if [ "$status" != "$expected" ]; then
    echo "FAIL [$status != $expected] $desc ($url)"
    ERRORS=$((ERRORS + 1))
  else
    echo "PASS [$status] $desc"
  fi
}

check "$BASE_URL/wp-json/wp/v2/posts?per_page=1" "200" "WordPress REST API reachable"
check "$BASE_URL/wp-json/" "200" "WP REST API root"
check "$BASE_URL/.well-known/healthcheck" "200" "Custom health endpoint" 2>/dev/null || \
  echo "SKIP health endpoint (not configured)"

exit $ERRORS
```

Save as `tools/operator/operator-smoke.sh`, commit alongside the feature code.

### k6 script (threshold validation)

```javascript
// operator-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 1,
  duration: '30s',
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],   // less than 1% failures
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://liveaccesshub.com';

export default function () {
  const res = http.get(`${BASE_URL}/wp-json/wp/v2/posts?per_page=1`);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 300ms': (r) => r.timings.duration < 300,
  });
  sleep(1);
}
```

Run: `k6 run operator-test.js -e BASE_URL=https://liveaccesshub.com`

### Structured operator test report

After running checks, append results to the continuity JSON or operator packet:

```json
{
  "operator_tests": {
    "gate": "9.5",
    "tool": "k6",
    "timestamp": "2026-07-23T14:00:00Z",
    "status": "PASS",
    "checks": [
      {"name": "WP REST API", "endpoint": "/wp-json/", "status": 200, "pass": true},
      {"name": "latency p95", "threshold": "<500ms", "actual": "120ms", "pass": true},
      {"name": "failure rate", "threshold": "<1%", "actual": "0%", "pass": true}
    ],
    "artifacts": ["tools/operator/operator-smoke.sh"]
  }
}
```

## Branch — k6 for performance validation

If the mission changes critical-path endpoints or adds new ones, load `llama-cpp` or any load-test skill to design k6 thresholds before running this gate.

## Reference: Tool Catalog

The full catalog of 20 operator-testing tools is documented in the session that produced this reference. Key tiers:

| Tier | Tools | Use when |
|------|-------|----------|
| Smoke | `smoke.sh`, `prodzilla`, `upright` | Quick post-merge confidence check |
| Synthetic | `checkly`, `grafana/synthetic-monitoring-agent` | Continuous monitoring between missions |
| Canary | `flagger`, `argo-rollouts` | Progressive rollout with auto-rollback |
| Real deps | `testcontainers`, `terratest` | Integration against real DB/services |
| Load | `k6`, `locust` | Performance validation under realistic traffic |
| Replay | `keploy` | Regression testing with recorded prod traffic |
