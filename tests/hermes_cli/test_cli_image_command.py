from pathlib import Path
from unittest.mock import patch

from cli import (
    HermesCLI,
    _collect_query_images,
    _format_image_attachment_badges,
)


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._attached_images = []
    return cli_obj


def _make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


class TestImageCommand:
    def test_handle_image_command_attaches_local_image(self, tmp_path):
        img = _make_image(tmp_path / "photo.png")
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            cli_obj._handle_image_command(f"/image {img}")

        assert cli_obj._attached_images == [img]


    def test_handle_image_command_bare_shows_usage(self):
        """Bare ``/image`` prints the usage hint through the CLI's real printing
        helpers — this branch raised NameError before the lazy import existed."""
        cli_obj = _make_cli()

        with patch("cli._cprint") as mock_print:
            cli_obj._handle_image_command("/image")

        assert cli_obj._attached_images == []
        rendered = " ".join(str(arg) for call in mock_print.call_args_list for arg in call.args)
        assert "Usage: /image <path>" in rendered

    def test_handle_image_command_rejects_non_image_file(self, tmp_path):
        file_path = tmp_path / "notes.txt"
        file_path.write_text("hello\n", encoding="utf-8")
        cli_obj = _make_cli()

        with patch("cli._cprint") as mock_print:
            cli_obj._handle_image_command(f"/image {file_path}")

        assert cli_obj._attached_images == []
        rendered = " ".join(str(arg) for call in mock_print.call_args_list for arg in call.args)
        assert "Not a supported image file" in rendered


class TestCollectQueryImages:
    def test_collect_query_images_accepts_explicit_image_arg(self, tmp_path):
        img = _make_image(tmp_path / "diagram.png")

        message, images = _collect_query_images("describe this", str(img))

        assert message == "describe this"
        assert images == [img]


    def test_collect_query_images_supports_tilde_paths(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        img = _make_image(home / "storage" / "shared" / "Pictures" / "cat.png")
        monkeypatch.setenv("HOME", str(home))
        # ntpath.expanduser ignores HOME (Python 3.8+) — it wants USERPROFILE.
        monkeypatch.setenv("USERPROFILE", str(home))

        message, images = _collect_query_images("describe this", "~/storage/shared/Pictures/cat.png")

        assert message == "describe this"
        assert images == [img]


class TestImageBadgeFormatting:
    def test_compact_badges_use_filename_on_narrow_terminals(self, tmp_path):
        img = _make_image(tmp_path / "Screenshot 2026-04-09 at 11.22.33 AM.png")

        badges = _format_image_attachment_badges([img], image_counter=1, width=40)

        assert badges.startswith("[📎 ")
        assert "Image #1" not in badges



def test_single_query_images_follow_realized_turn_route(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.cli_single_query import _route_single_query_images

    cli_obj = SimpleNamespace(
        provider="text-provider", requested_provider="text-provider", model="text-model",
        _preprocess_images_with_vision=lambda *_args, **_kwargs: "unexpected auxiliary analysis",
    )
    decisions = []
    monkeypatch.setattr(
        "agent.image_routing.decide_image_input_mode",
        lambda provider, model, _config, *, requested_provider: (
            decisions.append((provider, model, requested_provider)) or "native"
        ),
    )
    monkeypatch.setattr(
        "agent.image_routing.build_native_content_parts",
        lambda text, _paths, **_kwargs: (
            [{"type": "text", "text": text}, {"type": "image_url"}], []
        ),
    )

    routed = _route_single_query_images(
        cli_obj, "inspect", "inspect", [Path("image.png")], [],
        turn_route={
            "model": "vision-model",
            "runtime": {"provider": "vision-provider", "requested_provider": "vision-provider"},
        },
    )

    assert decisions == [("vision-provider", "vision-model", "vision-provider")]
    assert routed[-1]["type"] == "image_url"
