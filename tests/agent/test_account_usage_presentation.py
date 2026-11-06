"""Dense serialization boundaries for untrusted provider quota metadata."""
from datetime import UTC, datetime
import json

import pytest

from agent.account_usage import AccountUsageRow, AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_presentation import serialize_account_usage_snapshot


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), float("-inf"), "12.5"])
def test_projection_excludes_non_numeric_values_and_private_provider_metadata(invalid):
    """Catch bool/string/non-finite coercion and accidental raw or free-form field disclosure."""
    private = "private-provider-value"
    snapshot = AccountUsageSnapshot(
        provider="openai-codex", source=private, fetched_at=datetime(2030, 1, 1, tzinfo=UTC),
        title=private, plan="https://private.example.invalid", identity=private,
        details=(private,), raw={"account": private}, credits_balance=invalid,
        windows=(AccountUsageWindow(label=private, label_key="session", used_percent=invalid,
                                    limit=invalid, limit_remaining=invalid, detail=private,
                                    reset_interval=private),),
        rows=(AccountUsageRow("credits_balance", {"value": invalid}),
              AccountUsageRow("api_key_usage", {"total": 5, "daily": invalid, "url": private}),
              AccountUsageRow(private, {"value": 5})),
    )

    view = serialize_account_usage_snapshot(snapshot)

    assert view is not None
    assert view["credits_balance"] is None and view["plan"] is None
    window = view["windows"][0]
    assert window["used_percent"] is None and window["remaining_percent"] is None
    assert window["limit"] is None and window["limit_remaining"] is None
    assert window["reset_interval"] is None
    assert view["rows"] == [{"key": "api_key_usage", "args": {"total": 5}}]
    wire = json.dumps(view, allow_nan=False)
    assert private not in wire and "private.example.invalid" not in wire


@pytest.mark.parametrize("used, remaining", [(-25, 100), (0, 100), (37.5, 62.5), (100, 0), (125, 0)])
def test_projection_bounds_provider_percentages_without_inverting_remaining(used, remaining):
    """Catch negative/over-100 remaining allowances and a used-versus-remaining inversion."""
    snapshot = AccountUsageSnapshot(
        provider="anthropic", source="oauth_usage_api", fetched_at=datetime(2030, 1, 1, tzinfo=UTC),
        windows=(AccountUsageWindow(label="Current session", used_percent=used),),
    )

    view = serialize_account_usage_snapshot(snapshot)

    assert view["windows"][0]["remaining_percent"] == remaining
    assert view["windows"][0]["used_percent"] + remaining == 100
