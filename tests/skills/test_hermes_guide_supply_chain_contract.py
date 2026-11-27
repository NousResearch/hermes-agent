"""Offline contracts for executable examples in the bundled Hermes guide.

Only Markdown is inspected: no installer, package manager or service is run.
These checks enforce the example contract, not runtime approval or artifact trust.
"""
import ast
from pathlib import Path
import re
import unittest
from unittest.mock import patch

GUIDE = Path(__file__).resolve().parents[2] / "skills/autonomous-ai-agents/hermes-agent"


def fenced_blocks(text, language):
    return re.findall(r"```" + language + r"\s*\n(.*?)\n```", text, re.S)


def unpinned_launches(text):
    """Return npx/uvx package arguments without an explicit version constraint."""
    failures = []
    for block in fenced_blocks(text, "yaml"):
        command = None
        for line in block.splitlines():
            match = re.search(r'command:\s*"(npx|uvx)"', line)
            if match:
                command = match.group(1)
            match = re.search(r"args:\s*(\[.*\])", line)
            if not match or command is None:
                continue
            args = ast.literal_eval(match.group(1))
            packages = [arg for arg in args if not arg.startswith("-")]
            if packages:
                package = packages[0]
                version = (
                    package.split("==", 1)[1] if command == "uvx" and "==" in package
                    else package.rsplit("@", 1)[1] if command == "npx" and "@" in package.rsplit("/", 1)[-1]
                    else ""
                )
                # Keep launcher grammars separate. These are canonical example
                # spellings, not a resolver or an artifact-integrity guarantee.
                # SemVer 2.0.0: semver.org/spec/v2.0.0.html
                number = r"(?:0|[1-9][0-9]*)"
                prerelease = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
                npm_version = (
                    rf"{number}\.{number}\.{number}"
                    rf"(?:-{prerelease}(?:\.{prerelease})*)?"
                    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
                )
                # PyPA version-specifiers: epoch/release/pre/post/dev/local.
                # Python 1.2 is an exact release, unlike an npm partial range.
                python_version = (
                    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
                    r"(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?"
                    r"(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:\.[a-z0-9]+)*)?"
                )
                pattern = npm_version if command == "npx" else python_version
                pinned = version == "<reviewed-version>" or re.fullmatch(pattern, version)
                if not pinned:
                    failures.append((command, package))
            command = None
    return failures


class GuideSupplyChainContract(unittest.TestCase):
    def test_quick_start_does_not_pipe_remote_code_to_a_shell(self):
        text = (GUIDE / "SKILL.md").read_text(encoding="utf-8")
        quick = text.split("## Quick Start", 1)[1].split("## Key Paths", 1)[0]
        for block in fenced_blocks(quick, "bash"):
            self.assertIsNone(re.search(r"curl[^\n]*\|\s*(bash|sh)\b", block))

    def test_install_guidance_requires_review_and_approval(self):
        text = (GUIDE / "SKILL.md").read_text(encoding="utf-8").lower()
        quick = text.split("## quick start", 1)[1].split("## key paths", 1)[0]
        # Audit the documented boundaries, not merely shared vocabulary. This
        # remains an offline document contract, not an agent/runtime safety test.
        normalized = " ".join(quick.split())
        for requirement in (
            r"download\s+it\s+for\s+inspection",
            r"verify\s+its\s+digest\s+against\s+an\s+independently\s+trusted\s+value",
            r"digest\s+computed\s+from\s+the\s+same\s+untrusted\s+download\s+is\s+not\s+proof\s+of\s+authenticity",
        ):
            with self.subTest(boundary=requirement):
                self.assertRegex(normalized, requirement)
        for requirement in ("immutable", "digest", "approval", "do not execute"):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, quick)

    def test_mcp_launch_examples_pin_the_package(self):
        text = (GUIDE / "references/native-mcp.md").read_text(encoding="utf-8")
        self.assertEqual(unpinned_launches(text), [])

    def test_mcp_templates_are_not_presented_as_ready_to_run(self):
        text = (GUIDE / "references/native-mcp.md").read_text(encoding="utf-8").lower()
        for requirement in ("reviewed-version", "not ready to run", "approval", "credentials"):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_service_mutation_examples_require_approval(self):
        for name in ("troubleshooting.md", "webhooks.md"):
            with self.subTest(name=name):
                text = (GUIDE / "references" / name).read_text(encoding="utf-8").lower()
                self.assertIn("explicit approval", text)
                self.assertIn("existing work", text)
                normalized = " ".join(text.split())
                self.assertRegex(normalized, r"exact profile(?:,| and) service")
                if name == "webhooks.md":
                    self.assertIn("service and exposure scope", normalized)
                    self.assertIn(
                        "an enablement request is not authorization for public exposure or a restart",
                        normalized,
                    )

    def test_bare_and_scoped_npm_names_are_not_versions(self):
        for package in ("server-example", "@vendor/server-example"):
            with self.subTest(package=package):
                template = '```yaml\ncommand: "npx"\nargs: ["-y", "' + package + '"]\n```'
                self.assertEqual(unpinned_launches(template), [("npx", package)])

    def test_tags_and_version_ranges_are_not_immutable_versions(self):
        for command, package in (("npx", "@vendor/server@latest"),
                                 ("npx", "server@^1.2.3"),
                                 ("uvx", "server==latest")):
            with self.subTest(command=command, package=package):
                template = '```yaml\ncommand: "' + command + '"\nargs: ["' + package + '"]\n```'
                self.assertEqual(unpinned_launches(template), [(command, package)])

    def test_versioned_launches_accept_scoped_names(self):
        template = ('```yaml\ncommand: "npx"\nargs: ["-y", "@vendor/server-example@1.2.3"]\n'
                    'command: "uvx"\nargs: ["server-example==1.2.3"]\n```')
        self.assertEqual(unpinned_launches(template), [])

    def test_npm_partial_wildcard_and_malformed_versions_are_rejected(self):
        for version in ("1.2", "1.2.x", "1.2.*", "01.2.3", "1.2.3.4",
                        "1.2.3garbage", "1.2.3-", "1.2.3-01", "1.2.3+",
                        "1.2.3_foo", "1.2.3\u0661"):
            with self.subTest(version=version):
                package = "server@" + version
                text = '```yaml\ncommand: "npx"\nargs: ["' + package + '"]\n```'
                self.assertEqual(unpinned_launches(text), [("npx", package)])

    def test_npm_exact_prerelease_build_and_placeholder_are_accepted(self):
        for version in ("0.0.0", "1.2.3", "1.2.3-rc.1", "1.2.3-alpha.0",
                        "1.2.3+build.007", "1.2.3-rc.1+build.7", "<reviewed-version>"):
            with self.subTest(version=version):
                text = '```yaml\ncommand: "npx"\nargs: ["@vendor/server@' + version + '"]\n```'
                self.assertEqual(unpinned_launches(text), [])

    def test_python_exact_versions_are_not_npm_semver(self):
        for version in ("1", "1.2", "1.2.3", "1!2.0", "1.2rc1", "1.2.post1",
                        "1.2.dev1", "1.2.post1.dev2", "1.2+linux.1", "<reviewed-version>"):
            with self.subTest(version=version):
                text = '```yaml\ncommand: "uvx"\nargs: ["server==' + version + '"]\n```'
                self.assertEqual(unpinned_launches(text), [])

    def test_python_wildcards_ranges_and_malformed_versions_are_rejected(self):
        for version in ("1.2.*", "1.2.x", "1.2garbage", "1.2+", "1.2-rc.x",
                        "1.2.3_foo", ">=1.2", "1.2,!=1.2.1", "latest", "1.2.3\u0661"):
            with self.subTest(version=version):
                package = "server==" + version
                text = '```yaml\ncommand: "uvx"\nargs: ["' + package + '"]\n```'
                self.assertEqual(unpinned_launches(text), [("uvx", package)])

    def test_install_contract_detects_missing_trust_and_inspection_guidance(self):
        text = (GUIDE / "SKILL.md").read_text(encoding="utf-8")
        for clause in ("download it for inspection", "against an independently trusted\nvalue",
                       "A digest computed from the same untrusted download is\nnot proof of authenticity"):
            with self.subTest(clause=clause):
                self.assertIn(clause, text)
                weakened = text.replace(clause, "[guidance omitted]")
                result = unittest.TestResult()
                case = GuideSupplyChainContract("test_install_guidance_requires_review_and_approval")
                with patch.object(Path, "read_text", return_value=weakened):
                    case.run(result)
                self.assertEqual(result.errors, [])
                self.assertFalse(result.wasSuccessful(), "weakened guidance must be rejected")

    def test_service_contract_detects_missing_scope_and_exposure_boundaries(self):
        original_read = Path.read_text
        fixtures = (("troubleshooting.md", "exact profile and service"),
                    ("webhooks.md", "exact profile, service and exposure\nscope"),
                    ("webhooks.md", "An enablement request\nis not authorization for public exposure or a restart"))
        for name, clause in fixtures:
            with self.subTest(name=name, clause=clause):
                target = GUIDE / "references" / name
                text = target.read_text(encoding="utf-8")
                self.assertIn(clause, text)
                weakened = text.replace(clause, "[guidance omitted]")
                def read_fixture(path, *args, **kwargs):
                    return weakened if path == target else original_read(path, *args, **kwargs)
                result = unittest.TestResult()
                case = GuideSupplyChainContract("test_service_mutation_examples_require_approval")
                with patch.object(Path, "read_text", read_fixture):
                    case.run(result)
                self.assertEqual(result.errors, [])
                self.assertFalse(result.wasSuccessful(), "missing boundary must be rejected")

    def test_prose_and_read_only_examples_are_not_package_launches(self):
        text = 'npx is a launcher; do not use pip install.\n```bash\ncurl https://example.test/health\n```'
        self.assertEqual(unpinned_launches(text), [])


if __name__ == "__main__":
    unittest.main()
