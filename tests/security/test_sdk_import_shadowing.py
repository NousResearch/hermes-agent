"""Regression tests for SDK dependency-shadowing after module import."""

import subprocess
import sys
import textwrap


def test_sdk_imports_are_bound_before_later_sys_path_shadowing(tmp_path):
    """Planting SDK shadow modules after Hermes import must not affect SDK use."""
    sdk_dir = tmp_path / "sdk_dir"
    marker = tmp_path / "marker.txt"
    sdk_dir.mkdir()

    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        sdk_dir = Path({str(sdk_dir)!r})
        marker = Path({str(marker)!r})
        sys.path.insert(0, str(sdk_dir))

        def write_sdk(name, exported_name):
            (sdk_dir / f"{{name}}.py").write_text(
                "from pathlib import Path\\n"
                f"Path({str(marker)!r}).open('a').write('benign {{name}} import\\\\n')\\n"
                f"class {{exported_name}}:\\n"
                "    def __init__(self, *args, **kwargs): pass\\n",
                encoding="utf-8",
            )

        write_sdk("openai", "OpenAI")
        write_sdk("anthropic", "Anthropic")
        with (sdk_dir / "anthropic.py").open("a", encoding="utf-8") as fh:
            fh.write("class AnthropicBedrock:\\n    def __init__(self, *args, **kwargs): pass\\n")
        write_sdk("firecrawl", "Firecrawl")

        import agent.auxiliary_client as auxiliary_client
        import run_agent
        import agent.anthropic_adapter as anthropic_adapter
        import tools.web_tools as web_tools

        before_shadow = marker.read_text(encoding="utf-8")
        assert "benign openai import" in before_shadow
        assert "benign anthropic import" in before_shadow
        assert "benign firecrawl import" in before_shadow

        for name in ("openai", "anthropic", "firecrawl"):
            (sdk_dir / f"{{name}}.py").write_text(
                "from pathlib import Path\\n"
                f"Path({str(marker)!r}).open('a').write('MALICIOUS {{name}} EXECUTED\\\\n')\\n",
                encoding="utf-8",
            )

        auxiliary_client.OpenAI()
        run_agent.OpenAI()
        anthropic_adapter._get_anthropic_sdk().Anthropic()
        web_tools.Firecrawl()

        after_shadow = marker.read_text(encoding="utf-8")
        assert "MALICIOUS" not in after_shadow
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
