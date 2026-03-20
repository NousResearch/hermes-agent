from argparse import Namespace
from unittest.mock import patch

from hermes_cli.product_config import load_product_config
from hermes_cli.product_setup import run_product_setup_wizard, setup_product_network


def _make_product_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", False),
        section=overrides.get("section", None),
    )


def test_product_setup_model_section_syncs_model_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _fake_model_setup(config):
        config["model"] = {
            "provider": "custom",
            "base_url": "http://127.0.0.1:8080/v1",
            "default": "qwen3.5-9b-local",
            "api_mode": "chat_completions",
        }

    with (
        patch("hermes_cli.product_setup.is_interactive_stdin", return_value=True),
        patch("hermes_cli.product_setup.setup_model_provider", side_effect=_fake_model_setup),
    ):
        run_product_setup_wizard(_make_product_args(section="model"))

    product_config = load_product_config()
    assert product_config["models"]["default_route"] == {
        "provider": "custom",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "qwen3.5-9b-local",
        "api_mode": "chat_completions",
    }


def test_product_setup_tools_section_syncs_cli_toolsets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _fake_tools_setup(config, first_install=False):
        config["platform_toolsets"] = {"cli": ["web", "browser", "mynah-tier1"]}

    with (
        patch("hermes_cli.product_setup.is_interactive_stdin", return_value=True),
        patch("hermes_cli.product_setup.setup_tools", side_effect=_fake_tools_setup),
    ):
        run_product_setup_wizard(_make_product_args(section="tools"))

    product_config = load_product_config()
    assert product_config["tools"]["hermes_toolsets"] == ["web", "browser", "mynah-tier1"]


def test_product_setup_network_section_updates_public_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.product_setup.prompt", lambda *args, **kwargs: "officebox.local")

    setup_product_network()

    product_config = load_product_config()
    assert product_config["network"]["public_host"] == "officebox.local"


def test_product_setup_noninteractive_prints_guidance(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with patch("hermes_cli.product_setup.is_interactive_stdin", return_value=False):
        run_product_setup_wizard(_make_product_args())

    out = capsys.readouterr().out
    assert "hermes config set model.provider custom" in out

