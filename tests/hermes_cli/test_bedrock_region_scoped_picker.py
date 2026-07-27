"""Regression tests for #28156 — Bedrock picker must be region-scoped.

Geo-prefixed cross-region inference profiles (us.*, eu.*, apac.*, ...) only
route from endpoints in their own geography. Offering us.* profiles to an
eu-central-2 picker produces configs AWS rejects regardless of credentials.
"""

from hermes_cli.model_setup_flows import (
    BEDROCK_GEO_PREFIXES,
    bedrock_model_routable_from_region,
    bedrock_region_geo_prefix,
)


class TestRegionGeoPrefix:
    def test_known_geographies(self):
        assert bedrock_region_geo_prefix("us-east-1") == "us."
        assert bedrock_region_geo_prefix("eu-central-2") == "eu."
        assert bedrock_region_geo_prefix("ap-southeast-1") == "ap."
        assert bedrock_region_geo_prefix("ca-central-1") == "ca."
        assert bedrock_region_geo_prefix("sa-east-1") == "sa."
        assert bedrock_region_geo_prefix("me-south-1") == "me."
        assert bedrock_region_geo_prefix("af-south-1") == "af."

    def test_unknown_region_is_empty(self):
        assert bedrock_region_geo_prefix("") == ""
        assert bedrock_region_geo_prefix("moon-base-1") == ""


class TestRoutableFromRegion:
    def test_us_profile_not_offered_in_eu(self):
        assert not bedrock_model_routable_from_region(
            "us.anthropic.claude-sonnet-4-6", "eu-central-2"
        )

    def test_eu_profile_offered_in_eu(self):
        assert bedrock_model_routable_from_region(
            "eu.anthropic.claude-sonnet-4-6", "eu-central-2"
        )

    def test_global_profile_offered_everywhere(self):
        for region in ("eu-central-2", "us-east-1", "ap-southeast-1"):
            assert bedrock_model_routable_from_region(
                "global.anthropic.claude-sonnet-4-6", region
            )

    def test_bare_foundation_id_offered_everywhere(self):
        assert bedrock_model_routable_from_region(
            "anthropic.claude-3-sonnet-20240229-v1:0", "eu-central-2"
        )

    def test_apac_spellings_route_in_ap_regions(self):
        for prefix in ("ap.", "apac.", "jp."):
            assert bedrock_model_routable_from_region(
                f"{prefix}anthropic.claude-sonnet-4-6", "ap-northeast-1"
            )

    def test_eu_profile_not_offered_in_us(self):
        assert not bedrock_model_routable_from_region(
            "eu.anthropic.claude-sonnet-4-6", "us-east-1"
        )

    def test_unknown_region_hides_nothing(self):
        assert bedrock_model_routable_from_region(
            "us.anthropic.claude-sonnet-4-6", ""
        )


class TestGeoPrefixContract:
    def test_every_geo_prefix_maps_to_a_routable_region_or_is_alias(self):
        """Invariant: each geo prefix is either reachable from some region's
        geo mapping or an ap-family alias — no dead entries."""
        mapped = {
            bedrock_region_geo_prefix(r)
            for r in (
                "us-east-1", "eu-central-2", "ap-southeast-1",
                "ca-central-1", "sa-east-1", "me-south-1", "af-south-1",
            )
        }
        ap_aliases = {"apac.", "jp."}
        for prefix in BEDROCK_GEO_PREFIXES:
            assert prefix in mapped or prefix in ap_aliases
