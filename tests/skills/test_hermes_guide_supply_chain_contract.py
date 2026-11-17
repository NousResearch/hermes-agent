"""Offline contracts for executable examples in the bundled Hermes guide.

Only Markdown is inspected: no installer, package manager or service is run.
These checks enforce the example contract, not runtime approval or artifact trust.
"""
import ast
from pathlib import Path
import re
import unittest

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
                pinned = re.fullmatch(r"(?:\d+\.\d+(?:\.\d+)?[\w.+-]*|<reviewed-version>)", version)
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

    def test_prose_and_read_only_examples_are_not_package_launches(self):
        text = 'npx is a launcher; do not use pip install.\n```bash\ncurl https://example.test/health\n```'
        self.assertEqual(unpinned_launches(text), [])


if __name__ == "__main__":
    unittest.main()
