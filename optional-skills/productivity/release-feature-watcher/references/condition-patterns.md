# Condition Patterns

Use these patterns to convert vague requests into actionable watcher conditions.

## PR / issue

- "Tell me when it ships" -> PR merged + public release after merge.
- "Tell me when it merges" -> PR merged only.
- "Tell me when the bug is fixed" -> issue closed or linked PR merged + release after fix + release notes mention issue/fix when possible.

## Model/API provider

- "Tell me when model X is out" -> official API docs list model + pricing/limits are published.
- "Tell me when OpenRouter has model X" -> router model list includes model + price/context data available.
- "Tell me when feature X supports tools/vision/audio" -> official docs or API reference confirms capability.

## Changelog/docs

- "Watch the changelog for X" -> new changelog entry after baseline contains exact feature/fix phrase or close synonym.
- "Watch docs until Y changes" -> target page/section contains field/endpoint/value and evidence is strong enough for action.

## Package/version

- "Tell me when package reaches X" -> registry latest version satisfies semver.
- "Tell me when bug fix is in npm" -> registry version after baseline + changelog/release notes mention bug/fix.

## Default noise

Stay quiet unless the condition is met, the watcher cannot keep checking, or the user explicitly requested intermediate updates.
