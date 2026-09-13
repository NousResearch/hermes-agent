import unittest

from mission_control_operating_layer import (
    build_capability_manifest,
    build_consumer_contract,
    sanitize_atlas_health,
)


class OperatingLayerTests(unittest.TestCase):
  def test_capability_manifest_is_explicit_and_non_executing(self):
    result = build_capability_manifest("forge")
    self.assertFalse(result["execution_enabled"])
    names = {item["name"] for item in result["capabilities"]}
    self.assertIn("approvals", names)
    self.assertIn("sanitized_health", names)

  def test_forge_consumes_only_sanitized_atlas_health_and_fails_safe(self):
    result = build_consumer_contract(
        "forge",
        {"status": "healthy", "version": "0.20.5", "service": "forge"},
        atlas_health={
            "status": "degraded",
            "version": "0.20.5",
            "service": "atlas",
            "checks": {"gateway": "healthy", "configuration": "degraded", "telemetry": "healthy"},
            "tenant_id": "must-not-cross-boundary",
        },
    )
    self.assertEqual(result["dependencies"], {"atlas": "degraded"})
    self.assertEqual(result["workload_policy"], "fail_safe")
    self.assertNotIn("tenant_id", repr(result))


  def test_atlas_remains_independent_when_forge_is_degraded(self):
    result = build_consumer_contract(
        "atlas",
        {"status": "healthy", "version": "0.20.5", "service": "atlas"},
        forge_health={"status": "unavailable"},
    )
    self.assertEqual(result["dependencies"], {"forge": "unavailable"})
    self.assertEqual(result["workload_policy"], "monitor_and_protect")
    self.assertIn("recovery_protection", result["capabilities"])


  def test_paco_is_optional_and_not_a_runtime_dependency(self):
    result = build_consumer_contract(
        "forge", {"status": "healthy", "version": "0.20.5", "service": "forge"}
    )
    self.assertEqual(result["paco"], "disconnected")
    self.assertEqual(result["status"], "healthy")


  def test_atlas_projection_has_explicit_allowlist(self):
    result = sanitize_atlas_health({"status": "healthy", "version": "0.20.5", "service": "atlas", "secret": "x"})
    self.assertEqual(set(result), {"status", "version", "service", "checks"})
    self.assertTrue(all(value == "unknown" for value in result["checks"].values()))


  def test_unknown_role_fails_closed(self):
    with self.assertRaises(ValueError):
        build_consumer_contract("paco", {"status": "healthy"})


if __name__ == "__main__":
  unittest.main()
