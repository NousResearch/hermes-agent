"""Regression tests for command alias collisions."""

from collections import defaultdict
import unittest

from hermes_cli.commands import COMMAND_REGISTRY, gateway_help_lines, resolve_command


class TestCommandAliasCollisions(unittest.TestCase):
    def test_aliases_are_unique_across_commands(self):
        owners: dict[str, list[str]] = defaultdict(list)

        for cmd in COMMAND_REGISTRY:
            for alias in cmd.aliases:
                owners[alias].append(cmd.name)

        collisions = {alias: names for alias, names in owners.items() if len(names) > 1}
        self.assertEqual(collisions, {})

    def test_queue_help_does_not_advertise_quit_alias(self):
        queue_lines = [line for line in gateway_help_lines() if line.startswith("`/queue ")]

        self.assertEqual(len(queue_lines), 1)
        self.assertNotIn("`/q`", queue_lines[0])
        self.assertEqual(resolve_command("q").name, "quit")


if __name__ == "__main__":
    unittest.main()
