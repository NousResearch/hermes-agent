"""Validate the shape of a VeraData /rates response.

Pure logic, no network calls — used both by the agent (to sanity-check a
response before quoting a rate) and by the skill's test suite.
"""
from __future__ import annotations

VALID_COUNTRIES = {"CO", "MX", "BR", "CL", "PE", "AR"}


def validate_rates(data: dict, expected_country: str | None = None) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty when is_valid is True."""
    if not isinstance(data, dict):
        return False, "response is not a JSON object"

    country = data.get("country")
    if country not in VALID_COUNTRIES:
        return False, f"country must be one of {sorted(VALID_COUNTRIES)}, got {country!r}"

    if expected_country is not None and country != expected_country:
        return False, f"expected country {expected_country!r}, got {country!r}"

    # At least one numeric rate signal must be present besides 'country'.
    numeric_fields = {k: v for k, v in data.items() if k != "country" and isinstance(v, (int, float))}
    if not numeric_fields:
        return False, "no numeric rate signal found in response"

    return True, ""


if __name__ == "__main__":
    import json
    import sys

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"INVALID: could not parse JSON ({e})")
        sys.exit(1)

    ok, reason = validate_rates(payload)
    if ok:
        print("VALID")
        sys.exit(0)
    else:
        print(f"INVALID: {reason}")
        sys.exit(1)
