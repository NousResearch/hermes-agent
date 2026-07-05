"""Phase 3 of the /boomerang spec: the native CommandDef.

Verifies /boomerang resolves, surfaces in the gateway picker with an arg field,
and is not cli_only.
"""
from hermes_cli.commands import resolve_command


class TestBoomerangCommandDef:
    def test_boomerang_resolves(self):
        d = resolve_command("boomerang")
        assert d is not None
        assert d.name == "boomerang"

    def test_has_task_arg_hint_for_discord_picker(self):
        d = resolve_command("boomerang")
        assert d.args_hint == "<task>"

    def test_is_gateway_visible_not_cli_only(self):
        d = resolve_command("boomerang")
        assert not getattr(d, "cli_only", False)
