# Plan: Discord Tag Alias Correction

## Problem

Agents naturally tag users using `@{first_name}` format (e.g., `@Lachlan`), but Discord requires `<@{user_id}>` format (e.g., `<@556627489947123749>`) for proper mentions. This causes mentions to fail silently.

## Solution Overview

Add a configurable tag alias mapping in the Discord gateway that automatically converts `@{name}` patterns to `<@{user_id}>` format before messages are sent.

## Design

### Configuration Structure

Add `tag_aliases` to Discord platform config:

```yaml
platforms:
  discord:
    enabled: true
    token: "${DISCORD_BOT_TOKEN}"
    extra:
      tag_aliases:
        lachlan: "556627489947123749"
        marvin: "123456789012345678"
        rumpypumpy: "556627489947123749"
```

### Implementation Approach

**Minimal code changes** - modify only the Discord adapter's `format_message()` method:

1. **Read config**: Load `tag_aliases` from `PlatformConfig.extra`
2. **Pattern matching**: Use regex to find `@{alias}` patterns (case-insensitive)
3. **Replacement**: Convert to `<@{user_id}>` format
4. **Preserve existing**: Already-correct `<@{id}>` mentions pass through unchanged

### Code Changes

#### File: `plugins/platforms/discord/adapter.py`

**Change 1**: Import `re` module if not already present (line 51 shows it's already imported)

**Change 2**: Modify `__init__` to store tag aliases from config (around line 1100-1200):

```python
def __init__(
    self,
    config: PlatformConfig,
    # ... other params
):
    # ... existing init code ...
    
    # Load tag aliases from config
    self._tag_aliases: Dict[str, str] = {}
    if config.extra and "tag_aliases" in config.extra:
        aliases = config.extra["tag_aliases"]
        if isinstance(aliases, dict):
            # Normalize to lowercase for case-insensitive matching
            self._tag_aliases = {k.lower(): v for k, v in aliases.items()}
```

**Change 3**: Extend `format_message()` to apply tag correction (line 2873):

```python
def format_message(self, content: str) -> str:
    """
    Format message for Discord.
    
    Discord uses its own markdown variant.
    Also applies tag alias correction if configured.
    """
    # Apply tag alias correction
    if self._tag_aliases:
        content = self._apply_tag_aliases(content)
    
    return content


def _apply_tag_aliases(self, content: str) -> str:
    """Replace @alias mentions with <@user_id> format.
    
    Matches @alias patterns (case-insensitive) and replaces with
    Discord's native mention format <@user_id>.
    
    Already-correct <@123> mentions are preserved unchanged.
    """
    # Pattern to match @username but NOT <@123> (already correct format)
    # Word boundary ensures we don't match partial words
    # Case-insensitive flag for flexible matching
    pattern = r'@(\b\w+\b)(?![>\d]*>)'
    
    def replace_mention(match):
        username = match.group(1)
        lowercase_username = username.lower()
        if lowercase_username in self._tag_aliases:
            user_id = self._tag_aliases[lowercase_username]
            return f"<@{user_id}>"
        return match.group(0)  # Return original if no match
    
    return re.sub(pattern, replace_mention, content, flags=re.IGNORECASE)
```

### Regex Pattern Details

The pattern `@(\b\w+\b)(?![>\d]*>)`:
- `@` - literal @ symbol
- `(\b\w+\b)` - capture word characters (username) with word boundaries
- `(?![>\d]*>)` - negative lookahead to avoid matching `<@123>` patterns

This ensures:
- `@Lachlan` → `<@556627489947123749>` ✓
- `<@556627489947123749>` → unchanged ✓
- `@lachlan` → `<@556627489947123749>` ✓ (case-insensitive)

### Testing

#### Unit Tests

Add test file: `tests/platforms/discord/test_tag_aliases.py`

```python
import pytest
from plugins.platforms.discord.adapter import DiscordAdapter
from gateway.config import PlatformConfig

class TestTagAliasCorrection:
    
    def test_basic_alias_replacement(self):
        """Test basic @name to <@id> conversion"""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi <@556627489947123749>"
    
    def test_case_insensitive(self):
        """Test that matching is case-insensitive"""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @Lachlan and @LACHLAN")
        assert result == "Hi <@556627489947123749> and <@556627489947123749>"
    
    def test_preserve_correct_format(self):
        """Test that <@id> format is preserved"""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi <@556627489947123749>")
        assert result == "Hi <@556627489947123749>"
    
    def test_mixed_mentions(self):
        """Test mixing alias and correct format"""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan and <@556627489947123749>")
        assert result == "Hi <@556627489947123749> and <@556627489947123749>"
    
    def test_no_aliases_configured(self):
        """Test that messages pass through unchanged when no aliases configured"""
        config = PlatformConfig(enabled=True, extra={})
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"
    
    def test_unknown_alias_unchanged(self):
        """Test that unknown aliases are not modified"""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @unknown")
        assert result == "Hi @unknown"
```

#### Integration Tests

Add test to `tests/e2e/test_discord_adapter.py` if it exists, or create new file.

### Documentation

Update `website/docs/platforms/discord.md` (or create if doesn't exist):

```markdown
## Tag Alias Correction

Hermes Agent can automatically convert `@username` mentions to Discord's native `<@user_id>` format.

### Configuration

Add `tag_aliases` to your Discord platform config:

```yaml
platforms:
  discord:
    extra:
      tag_aliases:
        lachlan: "556627489947123749"
        marvin: "123456789012345678"
```

### How It Works

- Matches `@username` patterns (case-insensitive)
- Replaces with `<@user_id>` format
- Preserves already-correct `<@user_id>` mentions
- Unknown aliases pass through unchanged

### Getting User IDs

To get a Discord user ID:
1. Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
2. Right-click a user's name
3. Select "Copy User ID"
```

## Implementation Steps

1. ✅ Fork repository to `lc-hermes/hermes-agent`
2. ⏳ Create feature branch `feat/discord-tag-aliases`
3. ⏳ Implement code changes in `plugins/platforms/discord/adapter.py`
4. ⏳ Add unit tests in `tests/platforms/discord/test_tag_aliases.py`
5. ⏳ Add documentation
6. ⏳ Run test suite: `./scripts/run_tests.sh`
7. ⏳ Create PR with clear description

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Regex too aggressive | Thorough test coverage, negative lookahead prevents matching `<@id>` |
| Performance impact | Minimal - single regex pass, only when aliases configured |
| Breaking existing configs | Backward compatible - only activates when `tag_aliases` present |
| Case sensitivity issues | Normalize to lowercase for matching |

## Success Criteria

- ✅ Agent responses with `@lachlan` correctly tag user in Discord
- ✅ Existing `<@id>` mentions still work
- ✅ No performance degradation
- ✅ All tests pass
- ✅ Minimal code changes (single file modification)

## Next Steps

1. Review this plan
2. Approve implementation approach
3. Begin coding
