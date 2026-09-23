"""Regression tests for #28156 — Bedrock picker must be region-scoped.

Geo-prefixed cross-region inference profiles (us.*, eu.*, apac.*, ...) only
route from endpoints in their own geography. Offering us.* profiles to an
eu-central-2 picker produces configs AWS rejects regardless of credentials.
"""

from hermes_cli.model_setup_flows_bedrock import (
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




