"""GET /api/profiles must expose ``profile.yaml`` ``ui_meta``.

Bot Mode sections live in ``profile.yaml`` -> ``ui_meta['hermes-bots']``
(``sectionId`` / ``sectionName``). The Desktop's fleet enumeration
(``electron/connection-registry.ts`` -> ``profileMetadataFromApiProfiles``)
only attaches section metadata to a bot when the API response carries
``ui_meta``. Without it, every bot on a SECONDARY connection renders under
"Unassigned" even though its ``profile.yaml`` has the section — the active
connection looks fine, which is what makes the bug hard to spot.
"""

from pathlib import Path

from hermes_cli.profiles import ProfileInfo, read_profile_meta
from hermes_cli.web_routers.profiles import _profile_to_dict


def _write_profile(root: Path, name: str, *, section_id: str, section_name: str) -> Path:
    profile_dir = root / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "description: test bot\n"
        "ui_meta:\n"
        "  hermes-bots:\n"
        f"    title: {name}\n"
        f"    sectionId: {section_id}\n"
        f"    sectionName: {section_name}\n",
        encoding="utf-8",
    )
    return profile_dir


def test_read_profile_meta_returns_ui_meta(tmp_path: Path) -> None:
    profile_dir = _write_profile(tmp_path, "bot-a", section_id="jm-delivery",
                                 section_name="Jennifer Mobley - delivery")

    meta = read_profile_meta(profile_dir)

    assert meta["ui_meta"]["hermes-bots"]["sectionId"] == "jm-delivery"
    assert meta["ui_meta"]["hermes-bots"]["sectionName"] == "Jennifer Mobley - delivery"


def test_api_profiles_row_carries_ui_meta(tmp_path: Path) -> None:
    profile_dir = _write_profile(tmp_path, "bot-b", section_id="jm-delivery",
                                 section_name="Jennifer Mobley - delivery")
    meta = read_profile_meta(profile_dir)

    info = ProfileInfo(
        name="bot-b", path=profile_dir, is_default=False, gateway_running=False,
        model=None, provider=None, has_env=False, skill_count=0, alias_path=None,
        alias_name=None, distribution_name=None, distribution_version=None,
        distribution_source=None, **meta,
    )
    row = _profile_to_dict(info)

    assert row["ui_meta"]["hermes-bots"]["sectionId"] == "jm-delivery"
    assert row["ui_meta"]["hermes-bots"]["sectionName"] == "Jennifer Mobley - delivery"
    # the fields the endpoint already served must not regress
    assert row["name"] == "bot-b"
    assert row["description"] == "test bot"


def test_profile_without_ui_meta_still_serves_empty_mapping(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles" / "bare"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text("description: bare\n", encoding="utf-8")

    meta = read_profile_meta(profile_dir)
    info = ProfileInfo(
        name="bare", path=profile_dir, is_default=False, gateway_running=False,
        model=None, provider=None, has_env=False, skill_count=0, alias_path=None,
        alias_name=None, distribution_name=None, distribution_version=None,
        distribution_source=None, **meta,
    )

    assert meta["ui_meta"] == {}
    assert _profile_to_dict(info)["ui_meta"] == {}
