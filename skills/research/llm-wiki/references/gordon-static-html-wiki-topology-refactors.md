# Gordon Static HTML Wiki — Topology Refactor Pattern

Use this when a hub page starts duplicating analysis that also appears in child/deep-dive pages.

## Trigger
- User says a page has overlap/duplication with subpages.
- Hub page contains detailed market scans, vertical analysis, packaging, and idea buckets all in one place.
- Multiple child pages repeat the same thesis, implementation pattern, sources, or validation plan.

## Refactor Shape
1. Keep the root topic page as a concise portfolio/map page only.
2. Create an intermediate child hub for shared analysis when two or more children share the same thesis.
3. Move vertical-specific content into leaf/deep-dive pages under that intermediate hub.
4. Keep unrelated paths as siblings under the root topic page, not under the intermediate hub.
5. Preserve old URLs with static redirect pages.
6. Update scheduled jobs/cron prompts so automated updates target the right level.
7. Deploy and verify live pages with curl using browser UA + `wiki_auth` cookie.

## Example from 2026-05-10
Old shape:
- `/wiki/business-opportunities` had duplicated AI consulting notes, trend scans, consulting offers, geography, validation, and idea buckets.
- `/wiki/business-opportunities/manufacturing-semicap-workflow-automation`
- `/wiki/business-opportunities/ai-workflow-automation-local-smbs`

New shape:
- `/wiki/business-opportunities` — concise portfolio hub only.
- `/wiki/business-opportunities/ai-consulting-workflow-automation` — shared AI consulting thesis, market signals, packaging, geography, validation.
  - `/wiki/business-opportunities/ai-consulting-workflow-automation/manufacturing-semicap-workflow-automation` — semicap/manufacturing-specific vertical.
  - `/wiki/business-opportunities/ai-consulting-workflow-automation/ai-workflow-automation-local-smbs` — local SMB-specific vertical.
- `/wiki/business-opportunities/acquire-local-service-business` — separate buy-and-improve path.

## Verification Checklist
- Hub page contains navigation/ranking only, not detailed repeated analysis.
- Intermediate hub owns shared thesis/market/packaging content.
- Leaf pages link back to intermediate hub and contain only vertical-specific details.
- Old URLs redirect to new canonical URLs.
- Log page records the refactor.
- Any cron job that updates these pages has its prompt updated to preserve the topology.
