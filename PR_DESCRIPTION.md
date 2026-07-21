# feat: Add automatic model deprecation detection and handling system

## Problem

Users frequently encounter errors when selecting models that have been deprecated or removed by providers. The current system provides generic error messages like "Model was not found in this provider's model listing" without helpful guidance or alternatives.

### Specific Issues:
1. **Poor user experience**: Users get cryptic error messages when models are deprecated
2. **No guidance**: No suggestions for alternative models or replacements
3. **Manual updates**: Static model lists require manual updates when providers deprecate models
4. **Recurring errors**: Same deprecated models cause repeated errors for multiple users

## Solution

Implemented a comprehensive **Model Deprecation System** that automatically detects, tracks, and handles deprecated models with user-friendly guidance.

### Key Features:

#### 🔍 **Automatic Detection**
- Detects when models become unavailable from provider APIs
- Tracks consecutive failures to identify deprecation patterns
- Smart thresholding (3 failures within 24 hours) to avoid false positives

#### 💡 **User-Friendly Messages**
- Clear deprecation notices with emoji indicators (⚠️)
- Lists alternative working models
- Explains why the model was deprecated
- Provides actionable next steps

#### 🔄 **Auto-Redirect**
- Automatically redirects to recommended replacement models
- Prioritizes known working alternatives
- Maintains user workflow continuity

#### 📊 **Persistent Tracking**
- Local database (`~/.hermes/model_deprecations.json`) for detected deprecations
- Hardcoded knowledge base for common deprecations
- Distinguishes between pending and confirmed deprecations

### Example User Experience:

**Before:**
```
Model `deepseek-chat` was not found in this provider's model listing.
```

**After:**
```
⚠️  Model `deepseek-chat` has been deprecated. Auto-redirecting to `deepseek-v4-pro`.
⚠️  Model `deepseek-chat` has been deprecated. Please use `deepseek-v4-pro` instead. 
Alternative models: `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-reasoner`
Reason: DeepSeek API no longer supports the deepseek-chat model

✓ Switched to deepseek-v4-pro
```

## Implementation

### New Files:
- `hermes_cli/model_deprecation.py` - Core deprecation system (500+ lines)
- `tests/hermes_cli/test_model_deprecation.py` - Comprehensive test suite
- `docs/MODEL_DEPRECATION.md` - Complete documentation

### Modified Files:
- `hermes_cli/models.py` - Enhanced model validation with deprecation checks

### Key Components:

#### 1. **Deprecation Database**
```python
class ModelDeprecationDB:
    """Database for tracking model deprecations."""
    - Persistent storage in JSON format
    - CRUD operations for deprecation records
    - Query by status (pending/confirmed)
```

#### 2. **Availability Checker**
```python
class ModelAvailabilityChecker:
    """Check if models are available from their providers."""
    - Cached API checks for performance
    - Provider-specific availability logic
    - Error handling and fallbacks
```

#### 3. **Deprecation Detection**
```python
def record_model_failure(model_id, provider, error_message, ...):
    """Record a model failure for deprecation tracking."""
    - Automatic threshold detection
    - Integration with known deprecations
    - Status progression (pending → confirmed)
```

#### 4. **User-Facing API**
```python
def get_deprecation_message(model_id, provider) -> Optional[str]:
    """Get user-friendly deprecation message."""

def should_redirect_model(model_id, provider) -> Optional[str]:
    """Check if model should be auto-redirected."""
```

### Known Deprecations:

Pre-configured with common deprecations:
```python
_KNOWN_DEPRECATIONS = {
    "deepseek-chat": {
        "provider": "deepseek",
        "redirect_model": "deepseek-v4-pro",
        "suggested_alternatives": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"],
        "reason": "DeepSeek API no longer supports the deepseek-chat model",
    },
}
```

## Testing

### Test Coverage:
- ✅ Database operations (CRUD)
- ✅ Serialization/deserialization
- ✅ Deprecation detection logic
- ✅ Integration with existing validation
- ✅ User experience scenarios
- ✅ Error message enhancement
- ✅ Auto-redirect functionality

### Test Results:
```
============================================================
Model Deprecation System - Basic Tests
============================================================
✓ deepseek-chat deprecation info correct
✓ deprecation message format correct
✓ model redirect correct
✓ record created
✓ record retrieved
✓ record updated
✓ record deleted
✓ serialization to dict works
✓ deserialization from dict works
✓ all deprecations structure correct
✓ non-deprecated model returns None
✓ non-deprecated model has no message
✓ non-deprecated model has no redirect
============================================================
All tests passed! ✅
============================================================

============================================================
Model Deprecation System - Integration Tests
============================================================
✓ deepseek-chat is deprecated: deepseek-v4-pro
✓ deprecation message generated
✓ redirect suggestion: deepseek-v4-pro
✓ gpt-4 is not deprecated
✓ integration test completed
✓ Error message contains deprecation information
✓ Auto-redirect suggested: deepseek-v4-pro
✓ Correctly identified as deprecated, redirects to deepseek-v4-pro
✓ Correctly identified as not deprecated
============================================================
All integration tests passed! ✅
============================================================
```

## Benefits

### For Users:
- **Better error messages**: Clear explanations instead of cryptic errors
- **Automatic fixes**: Auto-redirect to working models
- **Guidance**: Specific alternative suggestions
- **Reduced frustration**: Fewer failed model selections

### For Maintainers:
- **Automatic detection**: Learns from user failures
- **Reduced support burden**: Self-service error resolution
- **Community knowledge**: Shared deprecation database
- **Data-driven insights**: Track deprecation patterns

### For the Project:
- **Improved reliability**: Proactive deprecation handling
- **Better UX**: Professional error handling
- **Future-proof**: Scalable for new providers
- **Competitive advantage**: Superior model management

## Migration & Compatibility

### Backward Compatible:
- ✅ Existing model validation still works
- ✅ No breaking changes to APIs
- ✅ Optional deprecation checks
- ✅ Graceful fallbacks

### Rollout:
- Feature is automatically enabled
- No configuration required
- Local database created on first use
- Known deprecations work immediately

## Future Enhancements

Potential improvements for future PRs:
- [ ] Community-driven deprecation database sync
- [ ] Automatic PR generation for static model updates
- [ ] Provider-specific deprecation policies
- [ ] Historical deprecation analytics dashboard
- [ ] Integration with models.dev for real-time data
- [ ] Webhook notifications for new deprecations

## Related Issues

- Addresses user frustration with deprecated models
- Improves upon #26269 (deepseek-chat removal) with a systematic solution
- Complements #26278 by providing automatic detection

## Checklist

- [x] Implementation complete
- [x] Tests passing
- [x] Documentation added
- [x] Backward compatible
- [x] No breaking changes
- [x] Code follows project style
- [x] Ready for review

---

**Summary**: This PR transforms model error handling from a source of frustration into a helpful, intelligent system that guides users to working alternatives while automatically learning from deprecation patterns.