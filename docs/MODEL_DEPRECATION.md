# Model Deprecation System

## Overview

The Model Deprecation System provides automatic detection and handling of deprecated or unavailable models from providers. This system improves user experience by:

1. **Detecting unavailable models** - Automatically identifies when models become unavailable
2. **Providing helpful error messages** - Shows users why a model doesn't work and suggests alternatives
3. **Auto-redirecting to replacements** - Automatically redirects to recommended replacement models
4. **Tracking deprecation patterns** - Learns from model failures to improve future suggestions

## Features

### 🔍 Automatic Detection

- **API-based detection**: Checks model availability against provider APIs
- **Failure tracking**: Records consecutive failures to identify deprecated models
- **Smart thresholding**: Only marks models as deprecated after multiple failures

### 💡 User-Friendly Messages

- **Clear deprecation notices**: Warns users when they try to use deprecated models
- **Alternative suggestions**: Provides list of working alternative models
- **Auto-redirect**: Automatically switches to recommended replacement models

### 📊 Persistent Tracking

- **Local database**: Stores deprecation records in `~/.hermes/model_deprecations.json`
- **Known deprecations**: Hardcoded knowledge base for common deprecations
- **Status tracking**: Distinguishes between pending and confirmed deprecations

## Usage

### Basic Example

When a user tries to use a deprecated model:

```bash
$ hermes model deepseek-chat
```

The system will respond with:

```
⚠️  Model `deepseek-chat` has been deprecated. Auto-redirecting to `deepseek-v4-pro`.
⚠️  Model `deepseek-chat` has been deprecated. Please use `deepseek-v4-pro` instead. 
Alternative models: `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-reasoner`
Reason: DeepSeek API no longer supports the deepseek-chat model

✓ Switched to deepseek-v4-pro
```

### Programmatic Usage

```python
from hermes_cli.model_deprecation import (
    get_deprecation_info,
    get_deprecation_message,
    should_redirect_model,
)

# Check if a model is deprecated
info = get_deprecation_info("deepseek-chat", "deepseek")
if info:
    print(f"Model is deprecated: {info['redirect_model']}")

# Get user-friendly message
message = get_deprecation_message("deepseek-chat", "deepseek")
print(message)

# Check for auto-redirect
redirect = should_redirect_model("deepseek-chat", "deepseek")
if redirect:
    print(f"Please use {redirect} instead")
```

## Architecture

### Components

1. **ModelDeprecationDB**: Database for tracking deprecation records
2. **ModelAvailabilityChecker**: Checks if models are available from providers
3. **Deprecation Detection**: Records failures and identifies patterns
4. **Integration Layer**: Enhances existing model validation with deprecation info

### Data Flow

```
User requests model → Model validation → Deprecation check → Enhanced response
                                      ↓
                              Check known deprecations
                                      ↓
                              Check detected deprecations
                                      ↓
                              Provide helpful message/redirect
```

## Configuration

### Known Deprecations

Known deprecations are hardcoded in `hermes_cli/model_deprecation.py`:

```python
_KNOWN_DEPRECATIONS: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "provider": "deepseek",
        "redirect_model": "deepseek-v4-pro",
        "suggested_alternatives": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"],
        "deprecation_date": "2025-04-01",
        "reason": "DeepSeek API no longer supports the deepseek-chat model",
    },
}
```

### Thresholds

Deprecation detection thresholds can be configured:

```python
_DEPRECATION_FAILURE_THRESHOLD = 3  # Number of failures before marking as deprecated
_DEPRECATION_FAILURE_WINDOW = 86400  # Time window for failure counting (24 hours)
_MODEL_AVAILABILITY_CACHE_TTL = 3600  # Cache duration (1 hour)
```

## API Reference

### Functions

#### `get_deprecation_info(model_id: str, provider: str) -> Optional[Dict[str, Any]]`

Get deprecation information for a model.

**Returns**: Dictionary with deprecation info or `None` if model is not deprecated

#### `get_deprecation_message(model_id: str, provider: str) -> Optional[str]`

Get a user-friendly deprecation message.

**Returns**: Formatted deprecation message or `None` if model is not deprecated

#### `should_redirect_model(model_id: str, provider: str) -> Optional[str]`

Check if a deprecated model should be redirected.

**Returns**: Target model ID for redirection or `None` if no redirect

#### `validate_model_with_deprecation_check(...)`

Extended model validation with deprecation checking.

**Returns**: Enhanced validation result with deprecation information

### Classes

#### `ModelDeprecationRecord`

Data structure for tracking model deprecations.

**Fields**:
- `model_id`: The model identifier
- `provider`: The provider name
- `first_detected`: When the deprecation was first detected
- `last_detected`: When the deprecation was last detected
- `failure_count`: Number of failures recorded
- `status`: "pending" or "confirmed"
- `suggested_alternatives`: List of alternative models
- `redirect_model`: Recommended replacement model
- `error_message`: Error message from the last failure

## Testing

### Run Tests

```bash
# Basic tests
python test_deprecation_basic.py

# Integration tests
python test_integration.py

# Full test suite
pytest tests/hermes_cli/test_model_deprecation.py -v
```

### Test Coverage

The system includes comprehensive tests for:
- Database operations (CRUD)
- Serialization/deserialization
- Deprecation detection logic
- Integration with existing validation
- User experience scenarios
- Error message enhancement

## Troubleshooting

### Clear Deprecation Cache

If you need to clear the deprecation database:

```python
from hermes_cli.model_deprecation import clear_deprecation_cache
clear_deprecation_cache()
```

### Debug Mode

To enable debug logging:

```python
import logging
logging.getLogger('hermes_cli.model_deprecation').setLevel(logging.DEBUG)
```

## Contributing

### Adding New Known Deprecations

To add a new known deprecation, edit `hermes_cli/model_deprecation.py`:

```python
_KNOWN_DEPRECATIONS["model-name"] = {
    "provider": "provider-name",
    "redirect_model": "replacement-model",
    "suggested_alternatives": ["alt-1", "alt-2"],
    "reason": "Explanation of why the model was deprecated",
}
```

### Improving Detection

To improve deprecation detection accuracy:

1. Adjust `_DEPRECATION_FAILURE_THRESHOLD` for sensitivity
2. Modify `_DEPRECATION_FAILURE_WINDOW` for time window
3. Enhance `ModelAvailabilityChecker._check_model_availability()` for better API checks

## Future Enhancements

- [ ] Community-driven deprecation database
- [ ] Automatic PR generation for updating static model lists
- [ ] Provider-specific deprecation policies
- [ ] Historical deprecation analytics
- [ ] Integration with models.dev for real-time deprecation data

## License

MIT License - See LICENSE file for details.