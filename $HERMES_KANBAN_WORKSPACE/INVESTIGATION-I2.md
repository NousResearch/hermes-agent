# Investigation I2: Google Chat tests — Platform.GOOGLE_CHAT missing (11 failures)

**Date:** 2026-05-08
**CI run:** 25574168674 — test (3.13)
**File:** `tests/gateway/test_google_chat.py` (2582 lines)

## Summary

`Platform.GOOGLE_CHAT` is not a built-in enum member. The `google_chat` platform exists only as a **bundled plugin** under `plugins/platforms/google_chat/`. The `Platform` enum's `_missing_()` method can dynamically create pseudo-members, but **only when called as `Platform("google_chat")`**. Attribute-style access (`Platform.GOOGLE_CHAT`) bypasses `_missing_()` entirely in Python's `Enum` implementation and raises `AttributeError`.

All 11 failing tests access `Platform.GOOGLE_CHAT` via attribute lookup before any `Platform("google_chat")` call has triggered `_missing_()`.

---

## Evidence

### 1. `Platform` enum does NOT list GOOGLE_CHAT as a built-in member

`gateway/config.py` lines 82-110:

```python
class Platform(Enum):
    LOCAL = "local"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    SIGNAL = "signal"
    MATTERMOST = "mattermost"
    MATRIX = "matrix"
    HOMEASSISTANT = "homeassistant"
    EMAIL = "email"
    SMS = "sms"
    DINGTALK = "dingtalk"
    API_SERVER = "api_server"
    WEBHOOK = "webhook"
    FEISHU = "feishu"
    WECOM = "wecom"
    WECOM_CALLBACK = "wecom_callback"
    WEIXIN = "weixin"
    BLUEBUBBLES = "bluebubbles"
    QQBOT = "qqbot"
    YUANBAO = "yuanbao"
    # --- no GOOGLE_CHAT ---
```

### 2. The `google_chat` plugin exists

```
plugins/platforms/google_chat/
├── __init__.py
├── adapter.py
├── oauth.py
└── plugin.yaml
```

Commit that added it: `44cd79e79` — `feat(plugins/google_chat): Google Chat platform adapter as a bundled plugin`

The `Platform._scan_bundled_plugin_platforms()` method scans `plugins/platforms/` and discovers `google_chat` as a bundled plugin. `_missing_("google_chat")` would successfully create a pseudo-member.

### 3. `_missing_()` creates pseudo-members on VALUE lookup, not ATTRIBUTE lookup

`gateway/config.py` lines 111-154:

```python
@classmethod
def _missing_(cls, value):
    """Accept unknown platform names only for known plugin adapters.
    Creates a pseudo-member cached in _value2member_map_ so that
    Platform("irc") is Platform("irc") holds True.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().lower()
    if value in cls._value2member_map_:
        return cls._value2member_map_[value]
    # ... scan bundled plugins ...
    if value in _Platform__bundled_plugin_names:
        pseudo = object.__new__(cls)
        pseudo._value_ = value
        pseudo._name_ = value.upper().replace("-", "_").replace(" ", "_")
        cls._value2member_map_[value] = pseudo
        cls._member_map_[pseudo._name_] = pseudo
        return pseudo
    # ... runtime registry check ...
```

Key point: `Platform.GOOGLE_CHAT` does **not** call `_missing_()`. Python's `Enum.__getattr__` does not delegate to `_missing_`; it raises `AttributeError` if the name is not in `_member_map_`. Only `Platform("google_chat")` invokes `_missing_()`.

### 4. All failing tests use `Platform.GOOGLE_CHAT` attribute access

Filing tests from `tests/gateway/test_google_chat.py`:

| Test class | Line | Failure pattern |
|---|---|---|
| `TestPlatformRegistration::test_enum_value` | 232 | `assert Platform.GOOGLE_CHAT.value == "google_chat"` |
| `TestEnvConfigLoading` (8 tests) | 266-325 | `cfg.platforms[Platform.GOOGLE_CHAT]` repeated |
| `TestAuthorizationEmailMatch` (3 tests) | 2453+ | `Platform.GOOGLE_CHAT` in allowlist logic |

None of these tests call `Platform("google_chat")` first. Therefore `_missing_()` is never triggered, `_member_map_` never gains "GOOGLE_CHAT", and attribute access fails with `AttributeError`.

---

## Root Cause

The tests assume `GOOGLE_CHAT` is a static built-in enum member, but it is actually a **dynamic plugin member**. The `Platform` enum's dynamic member mechanism (`_missing_()`) only works for value-based lookups (`Platform("google_chat")`), not attribute lookups (`Platform.GOOGLE_CHAT`).

When the tests import `Platform` from `gateway.config`, `GOOGLE_CHAT` is absent from `_member_map_`. Because no test calls `Platform("google_chat")` before accessing the attribute, the pseudo-member is never created, and all 11 tests fail.

---

## Fix Options

### Option A: Add `GOOGLE_CHAT` as a built-in enum member (recommended)

Add one line to `gateway/config.py` in the `Platform` enum:

```python
    YUANBAO = "yuanbao"
    GOOGLE_CHAT = "google_chat"
```

This treats Google Chat the same as all other built-in platforms. It is the simplest fix and aligns with the test assumptions. The plugin adapter under `plugins/platforms/google_chat/` will still work; the built-in enum member and the plugin are not mutually exclusive.

### Option B: Change tests to use `Platform("google_chat")`

Replace every `Platform.GOOGLE_CHAT` in the tests with `Platform("google_chat")`. This would trigger `_missing_()` and create the pseudo-member. However, this is more invasive (many changes across 2582 lines) and makes the tests inconsistent with how all other platforms are referenced.

### Option C: Force early `_missing_()` call in test setup

Add a fixture or module-level call to `Platform("google_chat")` before attribute access. Fragile and non-obvious.

---

## Recommendation

**Option A** — add `GOOGLE_CHAT = "google_chat"` to the `Platform` enum in `gateway/config.py`.

Rationale:
- The Google Chat adapter is a real, committed feature (`44cd79e79`).
- It has a full test suite (2582 lines) and plugin scaffolding.
- All other shipped platforms are built-in enum members; `google_chat` should be too.
- One-line fix vs. ~30+ line changes across test file.
- No risk to plugin loader — `_missing_()` will still work, and the built-in member takes precedence.

---

## Files to Modify (if fix is approved)

```
gateway/config.py   # add "GOOGLE_CHAT = \"google_chat\"" after YUANBAO
```

No test changes needed with Option A.
